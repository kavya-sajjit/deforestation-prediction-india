"""Central configuration for the v2 pipeline.

Every number that defines the task lives here, so the README, the paper-style
write-up and the code cannot drift apart.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Study regions. Each is a ~30 km x 30 km box (lon_min, lat_min, lon_max, lat_max).
# Use scripts/scan_hansen_aoi.py to move a box onto an area with enough forest
# loss; record any change you make here, because the README reports these boxes.
# ---------------------------------------------------------------------------
REGIONS: Dict[str, Dict] = {
    "western_ghats": dict(
        bbox=(75.75, 11.85, 76.05, 12.15),
        biome="Tropical evergreen / moist deciduous (Kodagu-Wayanad)",
    ),
    "odisha_mining": dict(
        bbox=(85.25, 21.85, 85.55, 22.15),
        biome="Sal forest with open-cast iron-ore mining (Keonjhar)",
    ),
    "northeast_hills": dict(
        bbox=(92.60, 22.80, 92.90, 23.10),
        biome="Subtropical hill forest with shifting cultivation (Mizoram)",
    ),
    "central_india": dict(
        bbox=(80.40, 21.90, 80.70, 22.20),
        biome="Mixed deciduous and teak forest (Balaghat)",
    ),
}

# ---------------------------------------------------------------------------
# Time windows. Dry-season (Jan-Apr) composites keep cloud cover low over India.
#   T0 = imagery the forecast model sees      (early 2019)
#   T1 = imagery after the target window ends (early 2024), used only for detection
# Hansen GFC "lossyear" codes: 1..23 = 2001..2023.
# ---------------------------------------------------------------------------
T0_WINDOW = "2019-01-01/2019-04-30"
T1_WINDOW = "2024-01-01/2024-04-30"
HISTORY_LOSS_YEARS = range(1, 19)        # 2001-2018: known before T0
AMBIGUOUS_LOSS_YEARS = range(19, 20)     # 2019: may overlap the T0 composite
TARGET_LOSS_YEARS = range(20, 24)        # 2020-2023: the forecast target
DETECT_LOSS_YEARS = range(19, 24)        # 2019-2023: between T0 and T1

HANSEN_VERSION = "GFC-2023-v1.11"
HANSEN_URL = ("https://storage.googleapis.com/earthenginepartners-hansen/"
              "{v}/Hansen_{v}_{layer}_{tile}.tif")
TREECOVER_THRESHOLD = 30                 # % canopy in 2000 to count as forest (GFW default)

# Sentinel-2 bands, in the order the models see them: R, G, B, NIR.
# (Pretrained B-cos ImageNet weights expect R, G, B first.)
S2_BANDS: List[str] = ["B04", "B03", "B02", "B08"]
BAND_NAMES: List[str] = ["red", "green", "blue", "nir"]
# Fixed reflectance scaling to [0, 1]. Fixed (not data-dependent) so the
# B-cos [x, 1-x] encoding means the same thing in every region.
BAND_SCALE: Dict[str, float] = {"B04": 0.25, "B03": 0.25, "B02": 0.25, "B08": 0.60}
MAX_CLOUD_COVER = 40                     # scene-level filter before pixel masking
MAX_SCENES = 12                          # least-cloudy scenes kept per window
SCL_VALID = [4, 5, 6, 7]                 # vegetation, bare soil, water, unclassified
BASELINE_04_DATE = "2022-01-25"          # L2A after this carry a +1000 DN offset

# ---------------------------------------------------------------------------
# Patches and labels
# ---------------------------------------------------------------------------
PATCH = 64                               # 64 px x 10 m = 640 m; NON-overlapping
MIN_FOREST_FRAC = 0.30                   # a patch must be >=30% forest at T0
MIN_VALID_FRAC = 0.95                    # cloud-free pixel share after compositing
POS_LOSS_FRAC = 0.02                     # >=2% of the patch lost  -> positive
NEG_LOSS_FRAC = 0.0                      # exactly no loss          -> negative
# patches in between are dropped as ambiguous rather than forced into a class
MAX_AMBIGUOUS_FRAC = 0.005               # forecast: drop patches with >0.5% loss in 2019
DENSITY_RADIUS_M = 2500                  # neighbourhood for "past loss nearby"

# ---------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------
BLOCK_PATCHES = 8                        # 8 x 8 patches = 5.12 km blocks
N_FOLDS = 5
BUFFER_PATCHES = 1                       # drop training patches touching a test block
SEED = 42


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr_scratch: float = 1e-3
    lr_pretrained: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8                    # early stopping on validation AP
    resize: int = 0                      # 0 = keep 64 px; pretrained runs use 128
    num_workers: int = 2
    seed: int = SEED
    extra: Dict = field(default_factory=dict)
