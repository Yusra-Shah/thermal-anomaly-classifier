"""Small MLP classifier with a scikit-learn style fit and predict_proba."""
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

SEED = 42


class MLPClassifier:
    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, batch_size=2048,
                 max_epochs=40, patience=5, val_size=0.15, seed=SEED):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.val_size, self.seed = val_size, seed

    def _net(self, n_in):
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        return nn.Sequential(*layers, nn.Linear(d, 1))

    def _scale(self, X):
        return torch.tensor((np.asarray(X, dtype=np.float32) - self.mean_) / self.std_)

    def fit(self, X, y, groups=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = X.mean(0)
        self.std_ = np.where(X.std(0) > 0, X.std(0), 1.0).astype(np.float32)

        # Validation split grouped by grid cell, used only for early stopping
        if groups is None:
            groups = np.arange(len(y))
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=self.val_size,
                                        random_state=self.seed).split(X, y, groups))
        Xt, yt = self._scale(X[tr]), torch.tensor(y[tr])
        Xv, yv = self._scale(X[va]), torch.tensor(y[va])

        pos_weight = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.net_ = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)

        best, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(len(yt), generator=gen)
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net_(Xt[idx]).squeeze(1), yt[idx]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = loss_fn(self.net_(Xv).squeeze(1), yv).item()
            if val < best - 1e-5:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.net_(self._scale(X)).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])
