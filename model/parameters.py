import torch

# ========== Dataset ==========
data_dir = "./data"
output_dir = "./results"
DATA_SET = "ADM"
X_RANGE = list(range(14))
Y_RANGE = list(range(2001))
GEOBOUNDARY = [0.3, 0.6, 1, 1.5, 0.1, 0.2, -0.7864, 0.7864]
NORMALIZE_INPUT = True
NORMALIZE_OUTPUT = True

# ========== Model ==========
MODEL_TYPE = "MLP"
LINEAR = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]
DROPOUT = 0.2
SKIP_CONNECTION = True
SKIP_HEAD = 1
PLOT_TRAINING_CURVE = True

# ========== Transformer Configuration ==========
HIDDEN_DIM = 512
TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 8
FF_DIM = 2048
DROPOUT_RATE = 0.2
USE_POSITIONAL_ENCODING = True

# ========== Training Hyperparameters ==========
OPTIM = "AdamW"  # Better regularization than Adam
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.9  # Slight decay to help generalization
REG_SCALE = 1e-4
BATCH_SIZE = 256
TRAIN_STEP = 500
EVAL_STEP = 1
EARLY_STOPPING_PATIENCE = 50
STOP_THRESHOLD = 1e-6  # Reasonable convergence check

# ========== Validation Split ==========
TEST_RATIO = 0.2

# ========== Reproducibility ==========
RAND_SEED = 1
torch.manual_seed(RAND_SEED)

# ========== Evaluation Settings ==========
USE_CPU_ONLY = False
MODEL_NAME = None
EVAL_MODEL = "ADM_paper"
NUM_COM_PLOT_TENSORBOARD = 1
