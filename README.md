# Post-Quantum Secure Federated Learning Framework

A comprehensive federated learning framework with support for Byzantine and data poisoning attack detection, post-quantum cryptography, and multiple defense mechanisms.

## Features

### Core Federated Learning
- **FedAvg Algorithm**: Standard federated averaging
- **CNN Model**: ResNet-style architecture for CIFAR-10 classification
- **Non-IID Data Distribution**: Dirichlet distribution (α=0.5) for realistic heterogeneous data
- **Multi-Client Setup**: Supports 10 clients with local training

### Security Features
- **Post-Quantum Cryptography**:
  - ML-KEM-768 (Kyber) for key encapsulation
  - ML-DSA-65 (Dilithium) for digital signatures
  - AES-256-GCM symmetric encryption for confidentiality and integrity
  - HKDF-SHA256 for key derivation
  - Secure update communication

- **Attack Detection & Defense**:
  - Byzantine Model Poisoning (configurable scale)
  - Data Poisoning via label flipping
  - Krum Defense (Byzantine-resilient aggregation)
  - Manhattan Distance Defense (Outlier detection)

### Evaluation
- Comprehensive metrics tracking
- Training loss monitoring
- Test accuracy evaluation
- Attack success rate measurement
- Timing analysis for cryptographic operations
- Communication overhead tracking

## Installation

### 1. Clone or Download the Repository
```bash
cd btp
```

### 2. Install Dependencies

#### Basic Requirements
```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scipy
```

#### Post-Quantum Cryptography (Optional)
For ML-KEM and ML-DSA support:
```bash
pip install liboqs-python
```

If `liboqs-python` is not available, the framework will use mock implementations for testing.

#### Complete Installation
```bash
pip install -r requirements.txt
```

### 3. Project Structure
```
btp/
├── basecode.py                  # Main entry point
├── config.py                    # Configuration settings
├── model.py                     # CNN architecture
├── dataset.py                   # CIFAR-10 data loading
├── pqc.py                       # Post-quantum cryptography
├── attacks.py                   # Attack implementations
├── defenses.py                  # Defense mechanisms
├── federated_learning.py        # FL framework
├── metrics.py                   # Evaluation metrics
├── experiments.py               # Experiment runners
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Usage

### 1. Run Single Experiment
```bash
# Experiment 1: Clean FL + FedAvg
python basecode.py --experiment 1

# Experiment 2: Byzantine Attack + FedAvg
python basecode.py --experiment 2

# Experiment 3: Byzantine Attack + Krum
python basecode.py --experiment 3

# Experiment 4: Data Poisoning + FedAvg
python basecode.py --experiment 4

# Experiment 5: Data Poisoning + Krum
python basecode.py --experiment 5

# Experiment 6: Byzantine Attack + PQC + FedAvg
python basecode.py --experiment 6

# Experiment 7: Byzantine Attack + PQC + Krum
python basecode.py --experiment 7

# Experiment 8: Byzantine Attack + Manhattan Distance
python basecode.py --experiment 8

# Experiment 9: Data Poisoning + Manhattan Distance
python basecode.py --experiment 9

# Experiment 10: Byzantine Attack + PQC + Manhattan Distance
python basecode.py --experiment 10
```

### 2. Run All Experiments
```bash
python basecode.py --all
```

### 3. Customize Experiments

Edit `config.py` to modify:
- Number of rounds, clients, local epochs
- Batch size and learning rate
- Byzantine attack parameters
- Data poisoning ratio
- PQC settings
- Defense mechanisms

Example:
```python
NUM_ROUNDS = 50              # Federated learning rounds
LOCAL_EPOCHS = 3             # Local training epochs
NUM_CLIENTS = 10             # Number of clients
BYZANTINE_SCALE = 3.0        # Poisoning scale factor
POISON_RATIO = 0.3           # Data poisoning ratio
DEFENSE_METHOD = 'krum'      # 'fedavg', 'krum', or 'manhattan'
```

## Experiments

### Experiment 1: Clean FL + FedAvg
Baseline federated learning with standard FedAvg aggregation.

**Expected Result**: Steady accuracy improvement, no attacks

### Experiment 2: Byzantine Attack + FedAvg
Federated learning with Byzantine model poisoning but no defense.

**Expected Result**: Accuracy degradation due to malicious updates

### Experiment 3: Byzantine Attack + Krum
Byzantine attack with Krum defense mechanism.

**Expected Result**: Improved robustness vs Exp 2

### Experiment 4: Data Poisoning + FedAvg
Data poisoning (label flipping) without defense.

**Expected Result**: Slow convergence and lower accuracy

### Experiment 5: Data Poisoning + Krum
Data poisoning with Krum defense.

**Expected Result**: Better accuracy than Exp 4

### Experiment 6: Byzantine Attack + PQC + FedAvg
Byzantine attack with ML-KEM encryption and ML-DSA signatures.

**Expected Result**: Secure communication, but vulnerable to poisoning without robust defense

### Experiment 7: Byzantine Attack + PQC + Krum
Secure setup with Krum defense.

**Expected Result**: Robust configuration against Byzantine attackers

### Experiment 8: Byzantine Attack + Manhattan Distance
Byzantine attack with Manhattan Distance defense mechanism.

**Expected Result**: Improved robustness by outlier detection

### Experiment 9: Data Poisoning + Manhattan Distance
Data poisoning with Manhattan Distance defense.

**Expected Result**: Better accuracy than Exp 4

### Experiment 10: Byzantine Attack + PQC + Manhattan Distance
Full secure setup with Manhattan Distance defense.

**Expected Result**: Most robust configuration with full encryption

## Output

### Results Directory
```
results/
├── Exp1_Clean_FL_summary.json
├── Exp1_Clean_FL_metrics.pkl
├── Exp1_Clean_FL_accuracy.png
├── Exp1_Clean_FL_loss.png
├── Exp1_Clean_FL_agg_time.png
├── Exp1_Clean_FL_updates.png
├── ... (for each experiment)
└── comparison_accuracy.png
```

### Metrics Tracked
- **Training Loss**: Average local training loss per round
- **Test Accuracy**: Global model accuracy on test set
- **Test Loss**: Global model loss on test set
- **Aggregation Time**: Time to aggregate updates
- **Encryption Time**: PQC encryption time (if enabled)
- **Decryption Time**: PQC decryption time (if enabled)
- **Signature Verification Time**: ML-DSA verification time
- **Valid/Invalid Updates**: Count of accepted/rejected updates
- **Suspicious Clients**: Krum/Manhattan Distance rejected/outlier count

## Configuration

### Key Parameters in `config.py`

#### Federated Learning
```python
NUM_ROUNDS = 50              # Total rounds
LOCAL_EPOCHS = 4             # Local training epochs per client
NUM_CLIENTS = 10             # Number of clients
BATCH_SIZE = 128             # Local batch size
LEARNING_RATE = 0.1          # SGD learning rate
```

#### Dataset
```python
DIRICHLET_ALPHA = 0.5        # Non-IID parameter (lower = more non-IID)
NUM_CLASSES = 10             # CIFAR-10 classes
```

#### Attacks
```python
BYZANTINE_SCALE = 3.0        # Poisoning multiplier
POISON_RATIO = 0.3           # Fraction of data to poison (0-1)
BYZANTINE_CLIENTS = 1        # Number of Byzantine clients
```

#### PQC
```python
PQC_ENABLED = True
ML_KEM_VARIANT = 'ML-KEM-768'
ML_DSA_VARIANT = 'ML-DSA-65'
SIGN_MESSAGE = True
ENCRYPT_MESSAGE = True
```

#### Defense
```python
DEFENSE_METHOD = 'fedavg'    # 'fedavg', 'krum', or 'manhattan'
KRUM_NUM_BYZANTINE = 1       # Expected number of Byzantine clients (f)
KRUM_MULTI_K = 5             # Number of updates to select in Multi-Krum (m)
MANHATTAN_DISTANCE_THRESHOLD = 0.5
MANHATTAN_DISTANCE_DEVIATION_FACTOR = 2.0
```

## Architecture Overview

```
┌─────────────────────────────────────┐
│         Global Model (Server)       │
└────────────────┬────────────────────┘
                 │
         ┌───────┼───────┐
         │       │       │
    ┌────v─┐ ┌──v───┐ ┌─v────┐
    │Client│ │Client│ │Client│  ...
    │  1   │ │  2   │ │  3   │
    └────┬─┘ └──┬───┘ └─┬────┘
         │      │       │
    ┌────v──────v───────v────┐
    │   Local Training        │
    │  (with optional attack) │
    └────┬──────┬───────┬────┘
         │      │       │
    ┌────v──────v───────v────┐
    │ Model Update Generation │
    └────┬──────┬───────┬────┘
         │      │       │
    ┌────v──────v───────v────┐
    │  Dilithium Signing      │
    │  Kyber Encryption       │
    └────┬──────┬───────┬────┘
         │      │       │
         └──────┼───────┘
                │
         ┌──────v──────────┐
         │  Server-side    │
         │ Verification    │
         │ Decryption      │
         └────────┬────────┘
                  │
         ┌────────v────────┐
         │  Defense Layer  │
         │ (FedAvg/Krum/   │
         │  Manhattan)     │
         └────────┬────────┘
                  │
         ┌────────v────────┐
         │  Model Agg.     │
         │  & Update       │
         └────────┬────────┘
                  │
              Repeat


## Extending the Framework

### Adding a New Defense

1. Create a defense class in `defenses.py`:
```python
class MyDefense:
    def aggregate(self, updates, client_ids=None):
        # Your defense logic
        return aggregated_update, stats
```

2. Register in `create_defense()`:
```python
elif defense_name == 'mydefense':
    return MyDefense()
```

3. Update `config.py`:
```python
DEFENSE_METHOD = 'mydefense'
```

### Adding a New Attack

1. Create an attack class in `attacks.py`:
```python
class MyAttack:
    def attack(self, update):
        # Your attack logic
        return poisoned_update
```

2. Integrate in `AttackManager`

### Modifying the Model

Edit `model.py` to change the CNN architecture:
```python
class CIFAR10CNN(nn.Module):
    def __init__(self):
        # Add/modify layers here
        pass
```

## Performance Tips

1. **GPU Acceleration**:
   ```python
   DEVICE = 'cuda'  # in config.py
   ```

2. **Reduce Data Size** (for testing):
   ```python
   NUM_ROUNDS = 5
   NUM_CLIENTS = 2
   ```

3. **Disable PQC** (for faster runs):
   ```python
   PQC_ENABLED = False
   ```

4. **Use Lightweight Defense**:
   ```python
   DEFENSE_METHOD = 'fedavg'
   ```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'oqs'`
**Solution**: PQC is optional. Either:
- Install liboqs: `pip install liboqs-python`
- Or disable PQC: Set `PQC_ENABLED = False` in config.py

### Issue: Out of Memory
**Solution**:
- Reduce `BATCH_SIZE`
- Reduce `NUM_CLIENTS`
- Set `DEVICE = 'cpu'`

### Issue: CIFAR-10 dataset not downloading
**Solution**:
```bash
mkdir -p ./data
# Dataset will be auto-downloaded on first run
```

## References

1. **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", ICML 2017
2. **Krum Defense**: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", NIPS 2017
3. **Post-Quantum Cryptography**: NIST PQC Standardization
4. **ML-KEM (Kyber)**: https://pq-crystals.org/kyber/
5. **ML-DSA (Dilithium)**: https://pq-crystals.org/dilithium/

## License

This project is provided for educational and research purposes.

## Citation

If you use this framework, please cite:
```bibtex
@software{pqcfl2024,
  title={Post-Quantum Secure Federated Learning Framework},
  author={Your Name},
  year={2024}
}
```

## Contact

For questions or issues, please reach out.
"# Federated_LeraningFramework_withPQC" 
