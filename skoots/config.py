from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Training config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()

_C.SYSTEM.NUM_GPUS = 2
_C.SYSTEM.NUM_CPUS = 1

# Define a BISM Model
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "bism_unext"
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.OUT_CHANNELS = 32
_C.MODEL.DIMS = [32, 64, 128, 64, 32]
_C.MODEL.DEPTHS = [2, 2, 2, 2, 2]
_C.MODEL.KERNEL_SIZE = 7
_C.MODEL.DROP_PATH_RATE = 0.0
_C.MODEL.LAYER_SCALE_INIT_VALUE = 1.0
_C.MODEL.ACTIVATION = "gelu"
_C.MODEL.BLOCK = "block3d"
_C.MODEL.CONCAT_BLOCK = "concatconv3d"
_C.MODEL.UPSAMPLE_BLOCK = "upsamplelayer3d"
_C.MODEL.NORMALIZATION = "layernorm"

# Training Configurations
_C.TRAIN = CN()
_C.TRAIN.DISTRIBUTED = True
_C.TRAIN.PRETRAINED_MODEL_PATH = [
    "/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models/Oct20_11-54-51_CHRISUBUNTU.trch"
]

# Loss function and their constructor keyowrds
_C.TRAIN.LOSS_EMBED = "tversky"
_C.TRAIN.LOSS_EMBED_KEYWORDS = ["alpha", "beta", "eps"]
_C.TRAIN.LOSS_EMBED_VALUES = [0.25, 0.75, 1e-8]

_C.TRAIN.LOSS_PROBABILITY = "tversky"
_C.TRAIN.LOSS_PROBABILITY_KEYWORDS = ["alpha", "beta", "eps"]
_C.TRAIN.LOSS_PROBABILITY_VALUES = [0.5, 0.5, 1e-8]

_C.TRAIN.LOSS_SKELETON = "tversky"
_C.TRAIN.LOSS_SKELETON_KEYWORDS = ["alpha", "beta", "eps"]
_C.TRAIN.LOSS_SKELETON_VALUES = [0.5, 1.5, 1e-8]

# We sum the loss values of each part together, scaled by some factor set here
_C.TRAIN.LOSS_EMBED_RELATIVE_WEIGHT = 1.0
_C.TRAIN.LOSS_PROBABILITY_RELATIVE_WEIGHT = 1.0
_C.TRAIN.LOSS_SKELETON_RELATIVE_WEIGHT = 1.0

# We may not consider each loss until a certain epoch. This may be
# because some tasks are hard to learn at the start, and must only be considered later
_C.TRAIN.LOSS_EMBED_START_EPOCH = -1
_C.TRAIN.LOSS_PROBABILITY_START_EPOCH = -1
_C.TRAIN.LOSS_SKELETON_START_EPOCH = 10

_C.TRAIN.TRAIN_DATA_DIR = []
_C.TRAIN.TRAIN_SAMPLE_PER_IMAGE = []
_C.TRAIN.TRAIN_BATCH_SIZE = 1
_C.TRAIN.VALIDATION_DATA_DIR = []
_C.TRAIN.VALIDATION_SAMPLE_PER_IMAGE = []
_C.TRAIN.VALIDATION_BATCH_SIZE = 1

_C.TRAIN.BACKGROUND_DATA_DIR = []
_C.TRAIN.BACKGROUND_SAMPLE_PER_IMAGE = []

_C.TRAIN.TRAIN_STORE_DATA_ON_GPU = []
_C.TRAIN.VALIDATION_STORE_DATA_ON_GPU = []
_C.TRAIN.BACKGROUND_STORE_DATA_ON_GPU = []

_C.TRAIN.STORE_DATA_ON_GPU = []
_C.TRAIN.INITIAL_SIGMA = [20.0, 20.0, 20.0]
_C.TRAIN.SIGMA_DECAY = [
    [0.66, 200],
    [0.66, 800],
    [0.66, 1500],
    [0.5, 20000],
    [0.5, 20000],
]
_C.TRAIN.NUM_EPOCHS = 10000
_C.TRAIN.LEARNING_RATE = 5e-4
_C.TRAIN.WEIGHT_DECAY = 1e-6
_C.TRAIN.OPTIMIZER = "adamw"  # Adam, AdamW, SGD,
_C.TRAIN.OPTIMIZER_KEYWORD_ARGUMENTS = []
_C.TRAIN.OPTIMIZER_KEYWORD_VALUES = []
_C.TRAIN.OPTIMIZER_EPS = 1e-8
_C.TRAIN.SCHEDULER = "cosine_annealing_warm_restarts"
_C.TRAIN.SCHEDULER_T0 = 10000 + 1
_C.TRAIN.MIXED_PRECISION = True
_C.TRAIN.N_WARMUP = 1500
_C.TRAIN.SAVE_PATH = (
    "/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models"
)
_C.TRAIN.SAVE_INTERVAL = 100
_C.TRAIN.CUDNN_BENCHMARK = True
_C.TRAIN.AUTOGRAD_PROFILE = False
_C.TRAIN.AUTOGRAD_EMIT_NVTX = False
_C.TRAIN.AUTOGRAD_DETECT_ANOMALY = False

# Augmentation
_C.AUGMENTATION = CN()
_C.AUGMENTATION.CROP_WIDTH = 300
_C.AUGMENTATION.CROP_HEIGHT = 300
_C.AUGMENTATION.CROP_DEPTH = 20
_C.AUGMENTATION.FLIP_RATE = 0.5
_C.AUGMENTATION.BRIGHTNESS_RATE = 0.4
_C.AUGMENTATION.BRIGHTNESS_RANGE = [-0.1, 0.1]
_C.AUGMENTATION.NOISE_GAMMA = 0.1
_C.AUGMENTATION.NOISE_RATE = 0.2
_C.AUGMENTATION.CONTRAST_RATE = 0.33
_C.AUGMENTATION.CONTRAST_RANGE = [0.75, 2.0]
_C.AUGMENTATION.AFFINE_RATE = 0.66
_C.AUGMENTATION.AFFINE_SCALE = [0.85, 1.1]
_C.AUGMENTATION.AFFINE_YAW = [-180, 180]
_C.AUGMENTATION.AFFINE_SHEAR = [-7, 7]
_C.AUGMENTATION.SMOOTH_SKELETON_KERNEL_SIZE = (3, 3, 1)
_C.AUGMENTATION.BAKE_SKELETON_ANISOTROPY = (1.0, 1.0, 3.0)
_C.AUGMENTATION.N_SKELETON_MASK_DILATE = 1

# Skoots Generics
_C.SKOOTS = CN()
_C.SKOOTS.VECTOR_SCALING = (60, 60, 60 // 5)
_C.SKOOTS.ANISOTROPY = (1.0, 1.0, 3.0)
_C.SKOOTS.NOTES = ''


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
