"""
Federated Learning Framework: Server, Client, and FedAvg Algorithm

Key fixes applied:
  1. compute_update: switched from get_flat_weights() to get_state_dict_flat()
     so BN running_mean/var buffers are included in the delta sent to server.
  2. update_global_model: switched from flat_weights to state_dict_flat so
     BN buffers are correctly aggregated and applied to the global model.
  3. local_train: added weight_decay, gradient clipping, cosine LR annealing.
  4. perform_round: global model kept in eval() during state_dict copy to
     clients; switched back to train() before local training.
  5. Weight-update norm is printed each round for easy debugging.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from model import create_model, CIFAR10CNN
from defenses import create_defense
from pqc import (PostQuantumCrypto, serialize_update, deserialize_update,
                  EncryptedUpdate, encrypt_update, decrypt_update,
                  serialize_state_dict, deserialize_state_dict)
from attacks import AttackManager
from trust import TrustManager
from config import *


class FLClient:
    """
    Federated Learning Client — responsible for local training.
    """

    def __init__(self, client_id: int, model: CIFAR10CNN,
                 dataset, attack_manager: Optional[AttackManager] = None,
                 device: str = DEVICE):
        """
        Args:
            client_id:      Unique client identifier
            model:          Local model (copy of global model)
            dataset:        Client's DataLoader
            attack_manager: Optional attack manager for Byzantine simulation
            device:         'cpu' or 'cuda'
        """
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.attack_manager = attack_manager
        self.device = device

        # Total rounds seen — used for cosine LR schedule
        self.round_num = 0

        # PQC keys — generate sig keypair whenever signing is enabled
        self.pqc = PostQuantumCrypto() if PQC_ENABLED else None
        if self.pqc and SIGN_MESSAGE:
            self.sig_pub_key, self.sig_sec_key = self.pqc.generate_sig_keypair()
        else:
            self.sig_pub_key, self.sig_sec_key = None, None

        self.metrics = {
            'local_loss': [],
            'training_time': [],
            'encryption_time': [],
            'signature_time': []
        }

    def local_train(self, num_epochs: int = LOCAL_EPOCHS,
                    learning_rate: float = LEARNING_RATE,
                    apply_data_poisoning: bool = False) -> Tuple[CIFAR10CNN, float]:
        """
        Perform local training on a fresh deep-copy of the global model.

        Changes vs original:
          - Deep-copy so we never mutate the global model accidentally
          - weight_decay added to SGD
          - Cosine annealing LR per epoch
          - Gradient clipping (GRAD_CLIP_NORM) prevents exploding updates
          - model set to train() before training, eval() before returning

        Returns:
            (trained_local_model, average_loss)
        """
        start_time = time.time()

        # ------------------------------------------------------------------
        # FIX: Deep-copy the ENTIRE state (parameters + BN buffers) so
        #      local training is fully isolated from the global model.
        # ------------------------------------------------------------------
        local_model = create_model(device=self.device)
        local_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        local_model.train()

        # Cosine-decay learning rate: starts at `learning_rate`, decays toward 0
        # over NUM_ROUNDS total rounds.
        cosine_lr = learning_rate * (
            0.5 * (1.0 + math.cos(math.pi * self.round_num / max(NUM_ROUNDS, 1)))
        )
        cosine_lr = max(cosine_lr, learning_rate * 0.01)   # floor at 1% of base LR

        optimizer = optim.SGD(
            local_model.parameters(),
            lr=cosine_lr,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,    # FIX: L2 regularisation
            nesterov=True                  # Nesterov momentum for better convergence
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            local_model.train()
            epoch_loss = 0.0

            for images, labels in self.dataset:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Apply data poisoning if enabled for this client
                if apply_data_poisoning and self.attack_manager and \
                   self.attack_manager.is_data_poisoning_client(self.client_id):
                    images, labels = self.attack_manager.apply_data_poisoning(images, labels)

                optimizer.zero_grad()

                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # ----------------------------------------------------------
                # FIX: Gradient clipping prevents a single bad batch from
                #      producing an enormous update that poisons the model.
                # ----------------------------------------------------------
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        training_time = time.time() - start_time
        average_loss = total_loss / (num_epochs * max(num_batches, 1))

        self.metrics['local_loss'].append(average_loss)
        self.metrics['training_time'].append(training_time)

        # Leave local model in eval mode — it won't be trained further
        local_model.eval()
        return local_model, average_loss

    def compute_update(self, local_model: CIFAR10CNN) -> torch.Tensor:
        """
        Compute model update delta = local_state_dict_flat - global_state_dict_flat.

        FIX (Critical): Previously used get_flat_weights() which only covered
        learnable parameters. Now uses get_state_dict_flat() which also includes
        BatchNorm running_mean / running_var buffers. Without this fix, BN
        stats were never synchronised and caused training divergence.
        """
        global_flat = self.model.get_state_dict_flat()
        local_flat = local_model.get_state_dict_flat()
        update = local_flat - global_flat
        return update

    def apply_attack(self, update: torch.Tensor) -> torch.Tensor:
        """
        Apply Byzantine attack if this is a designated Byzantine client.
        """
        if self.attack_manager and self.attack_manager.is_byzantine_client(self.client_id):
            return self.attack_manager.apply_model_poisoning(self.client_id, update)
        return update

    def encrypt_and_sign_update(self, update: torch.Tensor,
                                server_kem_pub_key: bytes = None) -> EncryptedUpdate:
        """
        Encrypt and sign a model update using PQC primitives.

        Security construction
        ---------------------
        1. KEM : ML-KEM-768 encapsulates a session key with the server's public key.
        2. KDF : The raw KEM shared secret is passed through HKDF-SHA256 to derive
                 a 256-bit AES key (done inside encrypt_update).
        3. AEAD: AES-256-GCM encrypts the payload; AAD = client_id ‖ round_num
                 binds the ciphertext to its context.
        4. Sign: ML-DSA-65 signs the *bound payload*:
                   client_id ‖ round_num ‖ kem_ciphertext ‖ nonce ‖ ciphertext
                 so that replaying or reordering messages from different clients
                 or rounds fails signature verification.
        """
        start_time = time.time()

        update_bytes    = serialize_update(update)
        round_num       = self.round_num

        # AAD for AES-GCM — authenticated but not encrypted
        aad = EncryptedUpdate.build_aad(self.client_id, round_num)

        signature       = None
        kem_ciphertext  = None
        encrypted_update = update_bytes
        nonce           = None

        # Step 1+2+3: KEM encapsulation → HKDF key derivation → AES-GCM encryption
        if self.pqc and ENCRYPT_MESSAGE and server_kem_pub_key:
            enc_start = time.time()
            kem_ciphertext, shared_secret = self.pqc.encapsulate(server_kem_pub_key)
            nonce, encrypted_update = encrypt_update(update_bytes, shared_secret, aad=aad)
            self.metrics['encryption_time'].append(time.time() - enc_start)

        # Step 4: Sign the full bound payload
        if self.pqc and SIGN_MESSAGE:
            sig_start = time.time()
            bound_msg = EncryptedUpdate.build_bound_message(
                self.client_id, round_num,
                kem_ciphertext, nonce, encrypted_update
            )
            signature = self.pqc.sign(bound_msg)
            self.metrics['signature_time'].append(time.time() - sig_start)

        return EncryptedUpdate(
            client_id=self.client_id,
            encrypted_update=encrypted_update,
            signature=signature,
            kem_ciphertext=kem_ciphertext,
            nonce=nonce,
            round_num=round_num,
        )

    @staticmethod
    def _xor_encrypt(data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption using shared secret (fallback)."""
        key_repeated = (key * ((len(data) // len(key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_repeated))

    def get_public_key(self) -> bytes:
        """Get client's ML-DSA signature public key."""
        return self.sig_pub_key


class FLServer:
    """
    Federated Learning Server — aggregation, decryption, and model distribution.
    """

    def __init__(self, model: CIFAR10CNN,
                 defense_name: str = DEFENSE_METHOD,
                 device: str = DEVICE,
                 trust_enabled: bool = None):
        """
        Args:
            model:         Global model
            defense_name:  'fedavg', 'krum', or 'manhattan'
            device:        Device string
            trust_enabled: Override TRUST_SCORE_ENABLED from config.
                           Defaults to the config value if None.
        """
        self.model = model
        self.device = device
        self.defense_name = defense_name

        # Trust scoring
        _trust_on = TRUST_SCORE_ENABLED if trust_enabled is None else trust_enabled
        self.trust_manager: Optional[TrustManager] = (
            TrustManager() if _trust_on else None
        )

        # Pass trust manager to the defense factory for attachment
        self.defense = create_defense(defense_name,
                                      trust_manager=self.trust_manager)

        # PQC for server (ML-KEM key pair for client-to-server encryption)
        self.pqc = PostQuantumCrypto() if PQC_ENABLED else None
        if self.pqc and ENCRYPT_MESSAGE:
            self.kem_pub_key, self.kem_sec_key = self.pqc.generate_kem_keypair()
        else:
            self.kem_pub_key, self.kem_sec_key = None, None

        self.client_public_keys = {}
        self.metrics = {
            'aggregation_time': [],
            'decryption_time': [],
            'verification_time': [],
            'num_valid_updates': [],
            'num_invalid_signatures': []
        }

    def register_client(self, client_id: int, public_key: bytes):
        """Register a client's ML-DSA public key for signature verification."""
        self.client_public_keys[client_id] = public_key

    def get_kem_public_key(self) -> bytes:
        """Return server's ML-KEM public key so clients can encrypt updates."""
        return self.kem_pub_key

    def verify_and_decrypt_updates(self,
                                   encrypted_updates: List[EncryptedUpdate],
                                   round_num: int = 0) \
            -> Tuple[List[torch.Tensor], List[int], Dict]:
        """
        Verify ML-DSA signatures and AES-GCM-decrypt each client update.

        Verification checks the *bound payload*:
          client_id ‖ round_num ‖ kem_ciphertext ‖ nonce ‖ ciphertext
        so any replay, reorder, or metadata-substitution attack is caught.

        AES-GCM decryption uses AAD = client_id ‖ round_num, which must
        match the AAD used during encryption or decryption raises InvalidTag.

        Returns:
            (decrypted_updates, valid_client_ids, stats)
        """
        start_time = time.time()

        decrypted_updates  = []
        valid_client_ids   = []
        num_valid          = 0
        num_invalid        = 0

        for enc_update in encrypted_updates:
            client_id      = enc_update.client_id
            encrypted_data = enc_update.encrypted_update
            signature      = enc_update.signature
            kem_ciphertext = enc_update.kem_ciphertext
            nonce          = enc_update.nonce
            msg_round      = enc_update.round_num

            # Build AAD and bound message using the round_num embedded in the message
            aad       = EncryptedUpdate.build_aad(client_id, msg_round)
            bound_msg = EncryptedUpdate.build_bound_message(
                client_id, msg_round, kem_ciphertext, nonce, encrypted_data
            )

            # ---- Step 1: Verify signature BEFORE decrypting ----
            # (fail-fast: don't waste compute on forged messages)
            is_valid = True
            if self.pqc and SIGN_MESSAGE and signature:
                ver_start = time.time()
                client_pub_key = self.client_public_keys.get(client_id)
                if client_pub_key:
                    is_valid = self.pqc.verify(bound_msg, signature, client_pub_key)
                self.metrics['verification_time'].append(time.time() - ver_start)

            if not is_valid:
                print(f"[Server] Invalid signature from client {client_id} "
                      f"(round {msg_round}) — dropping update.")
                num_invalid += 1
                continue

            # ---- Step 2: Decrypt ----
            decrypted_data = encrypted_data
            if self.pqc and ENCRYPT_MESSAGE and kem_ciphertext and nonce:
                dec_start = time.time()
                try:
                    shared_secret  = self.pqc.decapsulate(kem_ciphertext)
                    decrypted_data = decrypt_update(nonce, encrypted_data,
                                                   shared_secret, aad=aad)
                except Exception as e:
                    print(f"[Server] Decryption failed for client {client_id}: {e}")
                    num_invalid += 1
                    self.metrics['decryption_time'].append(time.time() - dec_start)
                    continue
                self.metrics['decryption_time'].append(time.time() - dec_start)

            # ---- Step 3: Deserialise ----
            try:
                update = deserialize_update(decrypted_data)
                decrypted_updates.append(update)
                valid_client_ids.append(client_id)
                num_valid += 1
            except Exception as e:
                print(f"[Server] Failed to deserialise update from client {client_id}: {e}")
                num_invalid += 1

        total_time = time.time() - start_time
        self.metrics['aggregation_time'].append(total_time)
        self.metrics['num_valid_updates'].append(num_valid)
        self.metrics['num_invalid_signatures'].append(num_invalid)

        stats = {
            'time':             total_time,
            'num_valid':        num_valid,
            'num_invalid':      num_invalid,
            'valid_client_ids': valid_client_ids,
        }

        return decrypted_updates, valid_client_ids, stats

    def aggregate_updates(self, updates: List[torch.Tensor],
                          valid_client_ids: List[int]) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate valid updates using the chosen defense mechanism.

        When trust scoring is enabled, the appropriate per-defense trust
        arguments are computed and passed in before the scores are refreshed
        with the result of this round's aggregation.

        Returns:
            (aggregated_update_tensor, aggregation_stats)
        """
        if not updates:
            return None, {}

        tm = self.trust_manager  # shorthand; may be None

        if self.defense_name == 'krum':
            modifiers = tm.get_krum_modifiers(valid_client_ids) if tm else None
            aggregated, stats = self.defense.aggregate(
                updates, valid_client_ids, trust_modifiers=modifiers
            )
        elif self.defense_name == 'manhattan':
            tw = tm.get_manhattan_weights(valid_client_ids) if tm else None
            aggregated, stats = self.defense.aggregate(
                updates, valid_client_ids, trust_weights=tw
            )
        else:  # fedavg
            tw = tm.get_weights(valid_client_ids) if tm else None
            aggregated = self.defense.aggregate(updates, trust_weights=tw)
            stats = {}

        # Update trust scores now that we have the aggregated reference
        if tm is not None and aggregated is not None:
            tm.update_scores(valid_client_ids, updates, aggregated)

        return aggregated, stats

    def get_trust_scores(self) -> Dict[int, float]:
        """
        Return current trust scores for all registered clients.
        Returns an empty dict when trust scoring is disabled.
        """
        if self.trust_manager is not None:
            return self.trust_manager.get_scores()
        return {}

    def update_global_model(self, aggregated_update: torch.Tensor,
                            round_num: int = 0) -> float:
        """
        Apply the aggregated update to the global model.

        FIX (Critical): Previously used get_flat_weights() / set_flat_weights()
        which silently dropped BatchNorm running_mean / running_var buffers.
        Now uses get_state_dict_flat() / set_state_dict_flat() to correctly
        update ALL tensors in the model, including BN statistics.

        SERVER_LEARNING_RATE is now 1.0 (standard FedAvg full replacement).

        Args:
            aggregated_update: Aggregated delta tensor (full state_dict_flat)
            round_num:         Current round (logged for debugging)

        Returns:
            float: L2 norm of the update (for debugging / monitoring)
        """
        if aggregated_update is None:
            return 0.0

        # Log update norm for debugging — should be non-zero every round
        update_norm = aggregated_update.norm(2).item()

        # Apply: new_global = old_global + SERVER_LEARNING_RATE * aggregated_delta
        current_flat = self.model.get_state_dict_flat()
        new_flat = current_flat + SERVER_LEARNING_RATE * aggregated_update

        # FIX: Use state_dict path (includes BN buffers)
        self.model.set_state_dict_flat(new_flat)

        return update_norm

    @staticmethod
    def _xor_decrypt(data: bytes, key: bytes) -> bytes:
        """Simple XOR decryption using shared secret (fallback)."""
        key_repeated = (key * ((len(data) // len(key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_repeated))


class FederatedLearner:
    """
    High-level Federated Learning orchestrator.
    Manages the entire FL process across rounds.
    """

    def __init__(self, clients: List[FLClient], server: FLServer):
        """
        Args:
            clients: List of FLClient instances
            server:  FLServer instance
        """
        self.clients = clients
        self.server = server
        self.num_clients = len(clients)
        self.history = {
            'loss': [],
            'accuracy': [],
            'update_norm': [],
            'attack_success_rate': []
        }

    def perform_round(self, round_num: int, test_loader=None,
                      apply_data_poisoning: bool = False,
                      apply_model_poisoning: bool = False) -> Dict:
        """
        Execute one full round of federated learning:
          1. Broadcast global model to all clients
          2. Each client trains locally
          3. Clients compute & encrypt their update delta
          4. Server decrypts & verifies updates
          5. Server aggregates with defense mechanism
          6. Server applies aggregated update to global model
          7. Evaluate global model on test set

        Returns:
            Dict of round statistics
        """
        round_stats = {
            'round': round_num,
            'client_losses': [],
            'test_loss': 0.0,
            'test_accuracy': 0.0,
            'update_norm': 0.0,
            'aggregation_stats': {},
            'pqc_stats': {},
            'trust_scores': {}
        }

        # ------------------------------------------------------------------ #
        # Step 1: Broadcast — copy global model state to all client models.   #
        # Use eval() on the global model during copy to ensure BN buffers     #
        # are in a stable, non-update state.                                  #
        # ------------------------------------------------------------------ #
        self.server.model.eval()
        global_state = copy.deepcopy(self.server.model.state_dict())
        for client in self.clients:
            client.model.load_state_dict(global_state)
            client.round_num = round_num   # Pass round number for cosine LR

        # ------------------------------------------------------------------ #
        # Step 2 & 3: Local training + compute update + encrypt              #
        # ------------------------------------------------------------------ #
        encrypted_updates = []
        for client in self.clients:
            # Local training (model set to train() internally)
            local_model, loss = client.local_train(
                apply_data_poisoning=apply_data_poisoning
            )
            round_stats['client_losses'].append(loss)

            # Compute full state_dict delta (includes BN buffers)
            update = client.compute_update(local_model)

            # Optionally corrupt the update (Byzantine attack)
            if apply_model_poisoning:
                update = client.apply_attack(update)

            # Encrypt and sign the update for secure transmission
            enc_update = client.encrypt_and_sign_update(
                update,
                self.server.get_kem_public_key()
            )
            encrypted_updates.append(enc_update)

        # ------------------------------------------------------------------ #
        # Step 4: Server — verify signatures & decrypt updates               #
        # ------------------------------------------------------------------ #
        updates, valid_client_ids, pqc_stats = self.server.verify_and_decrypt_updates(
            encrypted_updates, round_num=round_num
        )
        round_stats['pqc_stats'] = pqc_stats

        # ------------------------------------------------------------------ #
        # Step 5: Aggregate using defense mechanism                          #
        # ------------------------------------------------------------------ #
        aggregated_update, agg_stats = self.server.aggregate_updates(
            updates, valid_client_ids
        )
        round_stats['aggregation_stats'] = agg_stats

        # ------------------------------------------------------------------ #
        # Step 6: Apply aggregated update to global model                    #
        # ------------------------------------------------------------------ #
        update_norm = self.server.update_global_model(aggregated_update, round_num)
        round_stats['update_norm'] = update_norm
        self.history['update_norm'].append(update_norm)

        # Capture trust scores (empty dict if trust disabled)
        round_stats['trust_scores'] = self.server.get_trust_scores()

        if update_norm < 1e-8 and len(updates) > 0:
            print(f"  [WARNING] Round {round_num+1}: Update norm is near-zero ({update_norm:.2e}). "
                  f"Weights may not be updating!")

        # ------------------------------------------------------------------ #
        # Step 7: Evaluate global model on test set                          #
        # ------------------------------------------------------------------ #
        if test_loader:
            test_loss, test_accuracy = self.evaluate(test_loader)
            round_stats['test_loss'] = test_loss
            round_stats['test_accuracy'] = test_accuracy
            self.history['accuracy'].append(test_accuracy)
            self.history['loss'].append(test_loss)

        return round_stats

    def evaluate(self, test_loader) -> Tuple[float, float]:
        """
        Evaluate the global model on the test set.

        Returns:
            (average_loss, accuracy_percent)
        """
        self.server.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.server.device)
                labels = labels.to(self.server.device)

                outputs = self.server.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0

        return avg_loss, accuracy
