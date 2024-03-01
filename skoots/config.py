import os.path
import warnings

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
_C.MODEL.COMPILE = False

# Training Configurations
_C.TRAIN = CN()
_C.TRAIN.TARGET = 'skoots'  # should never be changed. only exists to allow bism interoperability.
_C.TRAIN.DISTRIBUTED = True
_C.TRAIN.PRETRAINED_MODEL_PATH = []
_C.TRAIN.LOAD_PRETRAINED_OPTIMIZER = False
_C.TRAIN.TRANSFORM_DEVICE = 'default' # or 'cpu'
_C.TRAIN.DATALOADER_OUTPUT_DEVICE = 'default' # or 'cpu'
_C.TRAIN.DATALOADER_NUM_WORKERS = 0
_C.TRAIN.DATALOADER_PREFETCH_FACTOR = 0


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
_C.TRAIN.SKELETON_MASK_RADIUS = 9
_C.TRAIN.SKELETON_MASK_FLANK_RADIUS= 3
_C.TRAIN.SAVE_INTERVAL = 100
_C.TRAIN.VALIDATE_EPOCH_SKIP = 10
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
_C.AUGMENTATION.ELASTIC_GRID_SHAPE = (6, 6, 2)  # integer tuple x, y, z
_C.AUGMENTATION.ELASTIC_GRID_MAGNITUDE = (0.05, 0.05, 0.01)  # small values are better here...

_C.AUGMENTATION.ELASTIC_RATE = 0.33

# Skoots Generics
_C.SKOOTS = CN()
_C.SKOOTS.VECTOR_SCALING = (60, 60, 60 // 5)
_C.SKOOTS.ANISOTROPY = (1.0, 1.0, 3.0)
_C.SKOOTS.NOTES = ''

_C.EXPERIMENTAL = CN()
_C.EXPERIMENTAL.DIST_THR = 10.0

# Experimental parameters to test the effect of background necessary to elicit high quality masks from sparse training
_C.EXPERIMENTAL.IS_SPARSE = False
_C.EXPERIMENTAL.SPARSE_BACKGROUND_PENALTY_MULTIPLIER = 10
_C.EXPERIMENTAL.BACKGROUND_N_ERODE = 0.0
_C.EXPERIMENTAL.BACKGROUND_SLICE_PERCENTAGE = 1.0

def _validate_model(cfg: CN):
    cm = cfg.MODEL

    assert cm.ARCHITECTURE == 'bism_unext', f'only "bism_unext" is supported for skoots training. Not {cm.ARCHITECTURE}'
    assert cm.IN_CHANNELS == 1, f'only greyscale input images currently supported. {cfg.MODEL.IN_CHANNELS=}!=1'
    assert cm.OUT_CHANNELS == cm.DIMS[-1], f'{cfg.MODEL.OUT_CHANNELS=} != {cfg.MODEL.DISM[-1]}'
    assert len(cm.DIMS) == len(cm.DEPTHS), f'must be same number of model DIMS as DEPTHS. len({cfg.MODEL.DIMS=}) != len({cfg.MODEL.DEPTS=})'
    assert cm.KERNEL_SIZE >= 3, f'Minimum model kernel size is 3, Kernel size of: {cfg.MODEL.KERNEL_SIZE=} is not supported.'
    if cm.KERNEL_SIZE >= 9: warnings.warn(f'Kernel size of {cfg.MODEL.KERNEL_SIZE=} is unusuially large. Model will be constructed normally, however may need large amounts of video memory.')
    assert cm.KERNEL_SIZE %2 == 1, f'Kernel size must be an odd number, not: {cfg.MODEL.KERNEL_SIZE=}.'

    if cm.DROP_PATH_RATE > 0.0:
        warnings.warn('drop path not tested and may lead to poor (or better?) results')
    if cm.LAYER_SCALE_INIT_VALUE != 1.0:
        warnings.warn('layer scale init value of anything other than 1 is not tested and may lead to poor (or better?) results')

    for val in [cm.BLOCK, cm.CONCAT_BLOCK, cm.UPSAMPLE_BLOCK]:
        assert '3d' in val, f'model part must be 3d not "{val=}"'

def _validate_training(_C: CN):
    ct = _C.TRAIN
    assert ct.TARGET == 'skoots', f'cfg.TRAIN.TARGET must be "skoots"'
    assert ct.DISTRIBUTED, 'cfg.TRAIN.DISTRIBUTED must equal True even with single GPU training.'

    for p in ct.PRETRAINED_MODEL_PATH:
        if p:
            assert os.path.exists(p), f'pretrained_model at {p} does not exist.'

    assert len(_C.TRAIN.LOSS_EMBED_KEYWORDS) == len(_C.TRAIN.LOSS_EMBED_VALUES), 'each embed loss fn keyword should have a value and vice versa'
    assert len(_C.TRAIN.LOSS_PROBABILITY_KEYWORDS) == len(_C.TRAIN.LOSS_PROBABILITY_VALUES), 'each prob loss fn keyword should have a value and vice versa'
    assert len(_C.TRAIN.LOSS_SKELETON_KEYWORDS) == len(_C.TRAIN.LOSS_SKELETON_VALUES), 'each skeleton loss fn keyword should have a value and vice versa'

    assert _C.TRAIN.LOSS_EMBED_RELATIVE_WEIGHT >= 0
    assert _C.TRAIN.LOSS_PROBABILITY_RELATIVE_WEIGHT>= 0
    assert _C.TRAIN.LOSS_SKELETON_RELATIVE_WEIGHT >= 0


    assert len(_C.TRAIN.TRAIN_DATA_DIR) == len(_C.TRAIN.TRAIN_SAMPLE_PER_IMAGE) == len(_C.TRAIN.TRAIN_STORE_DATA_ON_GPU), 'must specify for each data source, how many times to sample the image in that source, and if to store that data on the GPU'
    assert len(_C.TRAIN.VALIDATION_DATA_DIR) == len(_C.TRAIN.VALIDATION_SAMPLE_PER_IMAGE) == len(_C.TRAIN.VALIDATION_STORE_DATA_ON_GPU), 'must specify for each data source, how many times to sample the image in that source, and if to store that data on the GPU'
    assert len(_C.TRAIN.BACKGROUND_DATA_DIR) == len(_C.TRAIN.BACKGROUND_SAMPLE_PER_IMAGE) == len(_C.TRAIN.BACKGROUND_STORE_DATA_ON_GPU), 'must specify for each data source, how many times to sample the image in that source, and if to store that data on the GPU'

    assert _C.TRAIN.TRAIN_BATCH_SIZE >= 1
    assert _C.TRAIN.VALIDATION_BATCH_SIZE >= 1

    assert len(_C.TRAIN.OPTIMIZER_KEYWORD_ARGUMENTS) == len(_C.TRAIN.OPTIMIZER_KEYWORD_VALUES), 'must have a value for each optimizer keyword and vice versa'
    assert os.path.exists(_C.TRAIN.SAVE_PATH), f'specified path to save model does not exist. {_C.TRAIN.SAVE_PATH=}'
    assert _C.TRAIN.VALIDATE_EPOCH_SKIP >= 1, 'cannot skip negative numbers'


def _validate_skoots(_C: CN):

    x,y,z = _C.SKOOTS.VECTOR_SCALING
    if x < 5 or y < 5:
        warnings.warn(f'SKOOTS vector scaling set below a reasonable value. Is this intentional? Default vector scaling is (60, 60, 12)')

    x,y,z = _C.SKOOTS.ANISOTROPY
    if not any([x==1, y==1, z==1]):
        warnings.warn(f'skoots anisotropy should be relative. The default value is (1, 1, 3), denoting Z is spatially 3x larger than X and Y.')

def validate_cfg(cfg):
    _validate_model(cfg)
    _validate_skoots(cfg)
    _validate_training(cfg)

def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
