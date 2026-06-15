"""
Configuration settings for Post-Quantum Secure Federated Learning Framework
"""

# ==================== FEDERATED LEARNING ====================
NUM_ROUNDS = 10
LOCAL_EPOCHS = 4
NUM_CLIENTS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.1
OPTIMIZER = 'sgd'

# ==================== DATASET ====================
DATASET_NAME = 'cifar-10'
NUM_CLASSES = 10
DIRICHLET_ALPHA = 0.5
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
DEFENSE_METHOD = 'fedavg'  # 'fedavg' or 'foolsgold'

# FoolsGold Parameters
FOOLSGOLD_ENABLED = False
FOOLSGOLD_SIMILARITY_THRESHOLD = 0.5
FOOLSGOLD_HISTORY_SIZE = 10

# Manhattan Distance Parameters
MANHATTAN_DISTANCE_ENABLED = False
MANHATTAN_DISTANCE_THRESHOLD = 0.5
MANHATTAN_DISTANCE_DEVIATION_FACTOR = 2.0  # Multiplier for standard deviation

# ==================== SECURE AGGREGATION (Future)
SECURE_AGGREGATION_ENABLED = False
SECRET_SHARING_THRESHOLD = 5

# ==================== TRUST SCORE (Future)
TRUST_SCORE_ENABLED = False
TRUST_WINDOW = 5

# ==================== MANHATTAN DISTANCE (Future)
MANHATTAN_DISTANCE_ENABLED = False
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

# ==================== EXPERIMENTAL SETTINGS ====================
EXPERIMENTS = {
    'exp1': {
        'name': 'Clean FL + FedAvg',
        'byzantine_enabled': False,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False
    },
    'exp2': {
        'name': 'Byzantine Attack + FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False
    },
    'exp3': {
        'name': 'Byzantine Attack + FoolsGold',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'foolsgold',
        'pqc_enabled': False
    },
    'exp4': {
        'name': 'Data Poisoning + FedAvg',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'fedavg',
        'pqc_enabled': False
    },
    'exp5': {
        'name': 'Data Poisoning + FoolsGold',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'foolsgold',
        'pqc_enabled': False
    },
    'exp6': {
        'name': 'Byzantine Attack + PQC + FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': True
    },
    'exp7': {
        'name': 'Byzantine Attack + PQC + FoolsGold',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'foolsgold',
        'pqc_enabled': True
    },
    'exp8': {
        'name': 'Byzantine Attack + Manhattan Distance',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'manhattan',
        'pqc_enabled': False
    },
    'exp9': {
        'name': 'Data Poisoning + Manhattan Distance',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'manhattan',
        'pqc_enabled': False
    },
    'exp10': {
        'name': 'Byzantine Attack + PQC + Manhattan Distance',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'manhattan',
        'pqc_enabled': True
    }
}

# ==================== DEVICE ====================
DEVICE = 'cpu'  # Use 'cuda' if GPU available
DTYPE = 'float32'


# ==================== RANDOM SEED ====================