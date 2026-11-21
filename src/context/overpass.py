"""The OpenStreetMap query service (Overpass) through two public mirrors, tried in order.

A mirror that fails is remembered for 10 minutes (data/cache/overpass), so repeated clicks do not keep asking a service
that is down; when both mirrors are remembered as down the call fails at once. Between two tries there is a pause of
2 seconds. A failure carries only the exception class or the HTTP status, never an address.
"""
import time

from . import diskcache

MIRRORS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
DOWN_S = 600                      # a failed mirror is not asked again for 10 minutes
PAUSE_S = 2.0                     # between the first and the second mirror
SOURCE = "OpenStreetMap contributors (ODbL), via Overpass"


def _name(i):
    return f"down_{i}"


def query(ql, get=None, cache_dir=None, mirrors=MIRRORS, pause=PAUSE_S, sleep=time.sleep):
    """The JSON answer of an Overpass query from the first mirror that answers. Raises RuntimeError with a short
    reason when none does."""
    get = get or diskcache.http_get
    reasons, asked = [], 0
    for i, url in enumerate(mirrors):
        if diskcache.read("overpass", _name(i), ttl=DOWN_S, root=cache_dir):
            continue
        if asked:
            sleep(pause)
        asked += 1
        try:
            return get(url, params={"data": ql}).json()
        except Exception as e:
            why = str(e) if isinstance(e, RuntimeError) else type(e).__name__
            reasons.append(why)
            diskcache.write("overpass", _name(i), {"at": diskcache.now_text(), "why": why}, root=cache_dir)
    if not asked:
        raise RuntimeError("the OpenStreetMap query service did not answer a moment ago, not asked again for 10 minutes")
    raise RuntimeError("the OpenStreetMap query mirrors failed (" + ", ".join(reasons) + ")")
