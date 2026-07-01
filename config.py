"""
Configuration settings for Post-Quantum Secure Federated Learning Framework
"""

# ==================== FEDERATED LEARNING ====================
NUM_ROUNDS = 50              # Sufficient rounds for CIFAR-10 convergence
LOCAL_EPOCHS = 3             # Fewer local epochs per round reduces client drift
NUM_CLIENTS = 10
BATCH_SIZE = 64              # Smaller batch → better gradient estimates
LEARNING_RATE = 0.01         # Lower LR; combined with cosine decay in training loop
SERVER_LEARNING_RATE = 1.0   # FIX: was 0.1 — standard FedAvg uses 1.0 (full update)
OPTIMIZER = 'sgd'
WEIGHT_DECAY = 1e-4          # L2 regularization for client optimizers
GRAD_CLIP_NORM = 5.0         # Gradient clipping norm for stable training

# ==================== DATASET ====================
DATASET_NAME = 'cifar-10'
NUM_CLASSES = 10
DIRICHLET_ALPHA = 1.0        # FIX: was 0.5 — more balanced non-IID split
DATA_SPLIT_RATIO = (0.8, 0.2)  # (train, test)
RANDOM_SEED = 42

# ==================== MODEL ====================
MODEL_NAME = 'cifar_cnn'
INPUT_CHANNELS = 3
INPUT_SIZE = 32

# ==================== ATTACKS ====================
# Byzantine Model Poisoning
BYZANTINE_ENABLED = False
BYZANTINE_SCALE = 3.0
BYZANTINE_CLIENTS = 1
ATTACK_TYPE = 'model_poisoning'  # or 'data_poisoning'

# Data Poisoning
DATA_POISONING_ENABLED = False
POISON_RATIO = 0.3
LABEL_FLIP_TYPE = 'random'  # or 'specific'

# ==================== POST-QUANTUM CRYPTOGRAPHY ====================
PQC_ENABLED = True
ML_KEM_VARIANT = 'ML-KEM-768'  # Kyber-768
ML_DSA_VARIANT = 'ML-DSA-65'   # Dilithium-2
SIGN_MESSAGE = True
ENCRYPT_MESSAGE = True

# ==================== DEFENSES ====================
DEFENSE_METHOD = 'fedavg'  # 'fedavg', 'krum', or 'manhattan'

# Krum Parameters
KRUM_ENABLED = False
KRUM_NUM_BYZANTINE = 1     # Expected number of Byzantine clients (f)
KRUM_MULTI_K = 5           # Number of updates to select in Multi-Krum (m)

# Manhattan Distance Parameters
MANHATTAN_DISTANCE_ENABLED = False
MANHATTAN_DISTANCE_THRESHOLD = 0.5
MANHATTAN_DISTANCE_DEVIATION_FACTOR = 2.0  # Multiplier for standard deviation

# ==================== SECURE AGGREGATION (Future)
SECURE_AGGREGATION_ENABLED = False
SECRET_SHARING_THRESHOLD = 5

# ==================== TRUST SCORE ====================
TRUST_SCORE_ENABLED = True      # Enable trust-based aggregation weighting
TRUST_WINDOW = 5                # Sliding window of rounds for score history
TRUST_ALPHA = 0.8               # EMA decay: new_score = alpha*old + (1-alpha)*update
TRUST_MIN = 0.1                 # Floor score — no client is fully silenced
TRUST_NORM_PENALTY = 0.5        # Weight of update-norm penalty vs cosine similarity

# ==================== MANHATTAN DISTANCE ====================
MANHATTAN_THRESHOLD = 0.5

# ==================== EVALUATION ====================
TRACK_TRAINING_LOSS = True
TRACK_TEST_ACCURACY = True
TRACK_ATTACK_SUCCESS = True
TRACK_AGGREGATION_TIME = True
TRACK_ENCRYPTION_TIME = True
TRACK_DECRYPTION_TIME = True
TRACK_SIGNATURE_TIME = True
TRACK_COMMUNICATION_OVERHEAD = True

# ==================== LOGGING & STORAGE ====================
LOG_INTERVAL = 1  # Log every N rounds
SAVE_MODEL_INTERVAL = 5  # Save model every N rounds
RESULTS_DIR = './results'
LOG_DIR = './logs'
VERBOSE = True

# ==================== CHECKPOINTING ====================
CHECKPOINT_ENABLED = True           # Enable crash-recovery checkpoints
CHECKPOINT_INTERVAL = 5             # Save a checkpoint every N rounds
CHECKPOINT_DIR = './checkpoints'    # Root directory for checkpoint files
CHECKPOINT_KEEP_LAST = 3            # Max checkpoint files to retain per experiment
RESUME_FROM_CHECKPOINT = True       # Auto-resume if a checkpoint exists

# ==================== EXPERIMENTAL SETTINGS ====================
EXPERIMENTS = {
    'exp1': {
        'name': 'Clean FL + FedAvg',
        'byzantine_enabled': False,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False,
        'trust_enabled': False
    },
    'exp2': {
        'name': 'Byzantine Attack + FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False,
        'trust_enabled': False
    },
    'exp3': {
        'name': 'Byzantine Attack + Krum',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'krum',
        'pqc_enabled': False,
        'trust_enabled': False
    },
    'exp4': {
        'name': 'Data Poisoning + FedAvg',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'fedavg',
        'pqc_enabled': False,
        'trust_enabled': False
    },
    'exp5': {
        'name': 'Data Poisoning + Krum',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'krum',
        'pqc_enabled': False,
        'trust_enabled': False
    },
    'exp6': {
        'name': 'Byzantine Attack + PQC + FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': True,
        'trust_enabled': False
    },
    'exp7': {
        'name': 'Byzantine Attack + PQC + Krum',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'krum',
        'pqc_enabled': True,
        'trust_enabled': False
    },
    'exp8': {
        'name': 'Byzantine Attack + Manhattan Distance',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'manhattan',
        'pqc_enabled': False,
        'trust_enabled': False
    },
    'exp9': {
        'name': 'Data Poisoning + Manhattan Distance',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'manhattan',
        'pqc_enabled': False,
        'trust_enabled': False
    },
    'exp10': {
        'name': 'Byzantine Attack + PQC + Manhattan Distance',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'manhattan',
        'pqc_enabled': True,
        'trust_enabled': False
    },
    'exp11': {
        'name': 'Byzantine Attack + Trust-Based FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False,
        'trust_enabled': True
    },
    'exp12': {
        'name': 'Byzantine Attack + PQC + Trust-Based FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': True,
        'trust_enabled': True
    }
}

# ==================== DEVICE ====================
import torch as _torch
DEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'
del _torch
DTYPE = 'float32'


# ==================== RANDOM SEED ====================