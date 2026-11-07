from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Grid cell size in degrees used to decide that two detections are the same place.
# 0.01 degrees is about 1.1 km, wide enough to absorb pixel centroid jitter
# between overpasses of a 375 m sensor.
GRID_DEG = 0.01

# A location is called a static source when it is detected on at least this many
# distinct days. STATIC_WINDOW "month" counts days inside each calendar month and
# uses the best month (NASA's own rule is 16 days in a month). "total" counts days
# over the whole download and only suits short windows such as 7 days.
STATIC_MIN_DAYS = 16
STATIC_WINDOW = "month"

# Headline features the model is allowed to see. They all describe a single detection.
# Repeat counts and coordinates are deliberately excluded:
#  - repeat counts define the label, so using them would be leakage
#  - coordinates would let the model memorise places instead of learning signatures
FEATURES = [
    "bright_ti4",
    "bright_ti5",
    "bright_diff",
    "frp",
    "scan",
    "track",
    "pixel_area",
    "frp_per_area",
    "confidence_ord",
    "is_night",
]

# hour_utc is kept out of the headline setup because it acts as a location proxy.
# It is only added back for the ablation runs.
HOUR_FEATURE = "hour_utc"

TARGET = "is_static"
RANDOM_STATE = 42
