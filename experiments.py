"""
Experiments: All 7 experimental scenarios
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict

from model import create_model
from dataset import get_cifar10_dataset
from federated_learning import FLClient, FLServer, FederatedLearner
from attacks import AttackManager
from metrics import MetricsTracker, ComparisonAnalyzer
from checkpoint import CheckpointManager
from config import *


class ExperimentRunner:
    """Runs federated learning experiments"""
    
    def __init__(self, experiment_config: Dict):
        """
        Initialize experiment

        Args:
            experiment_config: Configuration for this experiment
        """
        self.config = experiment_config
        self.experiment_name = experiment_config['name']
        self.metrics_tracker = MetricsTracker(self.experiment_name)
        self.checkpoint_mgr  = CheckpointManager(self.experiment_name)

        print(f"\n{'='*60}")
        print(f"Initializing Experiment: {self.experiment_name}")
        print(f"{'='*60}")
    
    def setup(self):
        """
        Setup clients, server, and attack manager.

        Per-experiment pqc_enabled overrides the global PQC_ENABLED flag
        so that experiments with pqc_enabled=False truly disable PQC.
        trust_enabled overrides TRUST_SCORE_ENABLED similarly.
        """
        import config as _cfg

        # Override global PQC flag for this experiment
        _exp_pqc = self.config.get('pqc_enabled', False)
        _cfg.PQC_ENABLED    = _exp_pqc
        _cfg.ENCRYPT_MESSAGE = _exp_pqc
        _cfg.SIGN_MESSAGE    = _exp_pqc

        # Override global trust flag for this experiment
        _exp_trust = self.config.get('trust_enabled', False)
        _cfg.TRUST_SCORE_ENABLED = _exp_trust

        # Create global model
        global_model = create_model(device=DEVICE)

        # Load dataset
        print("Loading CIFAR-10 dataset...")
        dataset = get_cifar10_dataset()

        # Create attack manager
        apply_byzantine     = self.config.get('byzantine_enabled', False)
        apply_data_poisoning = self.config.get('data_poisoning_enabled', False)

        attack_manager = None
        if apply_byzantine or apply_data_poisoning:
            attack_manager = AttackManager(
                num_clients=NUM_CLIENTS,
                num_byzantine=BYZANTINE_CLIENTS,
                byzantine_scale=BYZANTINE_SCALE,
                poison_ratio=POISON_RATIO
            )
            byzantine_clients = attack_manager.get_byzantine_clients()
            print(f"Byzantine clients: {byzantine_clients}")

        # Create clients
        print(f"Creating {NUM_CLIENTS} clients...")
        clients = []
        for client_id in range(NUM_CLIENTS):
            client_model = create_model(device=DEVICE)
            client_model.load_state_dict(global_model.state_dict())

            client_dataset = dataset.get_client_dataloader(
                client_id,
                batch_size=BATCH_SIZE,
                train=True,
                shuffle=True
            )

            client = FLClient(
                client_id=client_id,
                model=client_model,
                dataset=client_dataset,
                attack_manager=attack_manager,
                device=DEVICE
            )
            clients.append(client)

        # Create server (uses updated PQC_ENABLED and TRUST_SCORE_ENABLED)
        defense_method = self.config.get('defense', 'fedavg')
        trust_enabled  = self.config.get('trust_enabled', False)
        server = FLServer(
            model=global_model,
            defense_name=defense_method,
            device=DEVICE,
            trust_enabled=trust_enabled
        )

        # Register client DSA public keys with the server
        for client in clients:
            server.register_client(client.client_id, client.get_public_key())

        # Create federated learner
        learner = FederatedLearner(clients, server)

        # Get test loader
        test_loader = dataset.get_global_testloader(batch_size=BATCH_SIZE)

        return learner, test_loader, dataset, attack_manager
    
    def run(self):
        """
        Run the experiment, with automatic checkpoint/resume support.

        Behaviour
        ---------
        - If RESUME_FROM_CHECKPOINT and a checkpoint exists, the global model
          and accumulated metrics are restored and training continues from
          `last_completed_round + 1`.
        - A checkpoint is written every CHECKPOINT_INTERVAL rounds AND after
          the very last round (guarantees the final state is always on disk).
        - Metrics are flushed to *_metrics_live.json after every round so
          that a crash loses at most one round of metric data.
        - On clean completion a DONE marker is written so subsequent runs
          can detect the experiment already finished.
        """
        # ── Setup infrastructure ──────────────────────────────────────────
        learner, test_loader, dataset, attack_manager = self.setup()

        apply_byzantine     = self.config.get('byzantine_enabled', False)
        apply_data_poisoning = self.config.get('data_poisoning_enabled', False)
        pqc_enabled         = self.config.get('pqc_enabled', False)
        trust_enabled       = self.config.get('trust_enabled', False)

        # ── Attempt to resume from checkpoint ────────────────────────────
        start_round = 0
        if RESUME_FROM_CHECKPOINT and CHECKPOINT_ENABLED:
            state = self.checkpoint_mgr.load_latest()
            if state:
                last = state['last_completed_round']
                # Restore model weights into the server's global model
                learner.server.model.load_state_dict(state['model_state_dict'])
                # Propagate to all clients so they start from the right point
                global_state = learner.server.model.state_dict()
                for client in learner.clients:
                    client.model.load_state_dict(global_state)
                    client.round_num = last + 1
                # Restore accumulated metrics
                self.metrics_tracker.restore(state.get('metrics', {}))
                start_round = last + 1
                print(f"  Resuming from round {start_round}/{NUM_ROUNDS}")
            else:
                print("  No checkpoint found - starting from scratch.")

        print(f"\nTraining Configuration:")
        print(f"  Rounds       : {start_round} -> {NUM_ROUNDS}")
        print(f"  Local Epochs : {LOCAL_EPOCHS}")
        print(f"  Batch Size   : {BATCH_SIZE}")
        print(f"  Byzantine    : {apply_byzantine}")
        print(f"  Data Poison  : {apply_data_poisoning}")
        print(f"  PQC Enabled  : {pqc_enabled}")
        print(f"  Trust Scoring: {trust_enabled}")
        print(f"  Defense      : {self.config.get('defense', 'fedavg')}")
        print(f"  Checkpointing: every {CHECKPOINT_INTERVAL} rounds "
              f"(dir: {self.checkpoint_mgr.ckpt_dir})")

        # ── Training loop ────────────────────────────────────────────────
        print("\nStarting training...")
        start_time = time.time()

        for round_num in range(start_round, NUM_ROUNDS):
            # ── Perform one FL round ──────────────────────────────────
            round_stats = learner.perform_round(
                round_num=round_num,
                test_loader=test_loader,
                apply_data_poisoning=apply_data_poisoning,
                apply_model_poisoning=apply_byzantine
            )

            # ── Track metrics ─────────────────────────────────────────
            self.metrics_tracker.add_round(round_num, round_stats)

            # ── Print progress ────────────────────────────────────────
            if (round_num + 1) % LOG_INTERVAL == 0:
                acc      = round_stats.get('test_accuracy', 0)
                loss     = round_stats.get('test_loss', 0)
                avg_loss = np.mean(round_stats.get('client_losses', [0]))
                print(f"  Round {round_num + 1:>4}/{NUM_ROUNDS} | "
                      f"TrainLoss: {avg_loss:.4f} | "
                      f"TestLoss: {loss:.4f} | "
                      f"Accuracy: {acc:.2f}%")

                # Print trust scores when enabled
                if trust_enabled and learner.server.trust_manager is not None:
                    learner.server.trust_manager.print_scores(round_num)

            # ── Incremental metrics flush (crash safety) ──────────────
            self.metrics_tracker.save_incremental()

            # ── Checkpoint ────────────────────────────────────────────
            is_last  = (round_num == NUM_ROUNDS - 1)
            is_ckpt  = CHECKPOINT_ENABLED and (
                (round_num + 1) % CHECKPOINT_INTERVAL == 0 or is_last
            )
            if is_ckpt:
                self.checkpoint_mgr.save(
                    round_num=round_num,
                    model_state_dict=learner.server.model.state_dict(),
                    metrics=self.metrics_tracker.metrics,
                    experiment_config=self.config,
                )

        total_time = time.time() - start_time

        # ── End-of-experiment summary & final saves ───────────────────
        print(f"\n{'='*60}")
        print(f"Experiment completed in {total_time:.2f}s")
        print(f"{'='*60}")

        self.metrics_tracker.print_summary()

        # Final full saves (replace live files with permanent versions)
        self.metrics_tracker.save_metrics()
        self.metrics_tracker.plot_results()

        # Mark experiment as successfully finished
        if CHECKPOINT_ENABLED:
            self.checkpoint_mgr.mark_complete()

        return self.metrics_tracker


def run_experiment_1():
    """Experiment 1: Clean FL + FedAvg"""
    config = {
        'name': 'Exp1: Clean FL + FedAvg',
        'byzantine_enabled': False,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_2():
    """Experiment 2: Byzantine Attack + FedAvg"""
    config = {
        'name': 'Exp2: Byzantine Attack + FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_3():
    """Experiment 3: Byzantine Attack + Krum"""
    config = {
        'name': 'Exp3: Byzantine Attack + Krum',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'krum',
        'pqc_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_4():
    """Experiment 4: Data Poisoning + FedAvg"""
    config = {
        'name': 'Exp4: Data Poisoning + FedAvg',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'fedavg',
        'pqc_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_5():
    """Experiment 5: Data Poisoning + Krum"""
    config = {
        'name': 'Exp5: Data Poisoning + Krum',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'krum',
        'pqc_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_6():
    """Experiment 6: Byzantine Attack + PQC + FedAvg"""
    config = {
        'name': 'Exp6: Byzantine Attack + PQC + FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': True
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_7():
    """Experiment 7: Byzantine Attack + PQC + Krum"""
    config = {
        'name': 'Exp7: Byzantine Attack + PQC + Krum',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'krum',
        'pqc_enabled': True
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_8():
    """Experiment 8: Byzantine Attack + Manhattan Distance"""
    config = {
        'name': 'Exp8: Byzantine Attack + Manhattan Distance',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'manhattan',
        'pqc_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_9():
    """Experiment 9: Data Poisoning + Manhattan Distance"""
    config = {
        'name': 'Exp9: Data Poisoning + Manhattan Distance',
        'byzantine_enabled': False,
        'data_poisoning_enabled': True,
        'defense': 'manhattan',
        'pqc_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_10():
    """Experiment 10: Byzantine Attack + PQC + Manhattan Distance"""
    config = {
        'name': 'Exp10: Byzantine Attack + PQC + Manhattan Distance',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'manhattan',
        'pqc_enabled': True,
        'trust_enabled': False
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_11():
    """Experiment 11: Byzantine Attack + Trust-Based FedAvg"""
    config = {
        'name': 'Exp11: Byzantine Attack + Trust-Based FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': False,
        'trust_enabled': True
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_experiment_12():
    """Experiment 12: Byzantine Attack + PQC + Trust-Based FedAvg"""
    config = {
        'name': 'Exp12: Byzantine Attack + PQC + Trust-Based FedAvg',
        'byzantine_enabled': True,
        'data_poisoning_enabled': False,
        'defense': 'fedavg',
        'pqc_enabled': True,
        'trust_enabled': True
    }
    runner = ExperimentRunner(config)
    return runner.run()


def run_all_experiments():
    """Run all 10 experiments"""
    print("\n" + "="*80)
    print("RUNNING ALL EXPERIMENTS FOR POST-QUANTUM SECURE FEDERATED LEARNING")
    print("="*80)
    
    all_metrics = {}
    
    experiments = [
        ("Exp1",  run_experiment_1),
        ("Exp2",  run_experiment_2),
        ("Exp3",  run_experiment_3),
        ("Exp4",  run_experiment_4),
        ("Exp5",  run_experiment_5),
        ("Exp6",  run_experiment_6),
        ("Exp7",  run_experiment_7),
        ("Exp8",  run_experiment_8),
        ("Exp9",  run_experiment_9),
        ("Exp10", run_experiment_10),
        ("Exp11", run_experiment_11),
        ("Exp12", run_experiment_12),
    ]
    
    for exp_name, exp_func in experiments:
        try:
            print(f"\n\nRunning {exp_name}...")
            metrics = exp_func()
            all_metrics[exp_name] = metrics.metrics
        except Exception as e:
            print(f"Error running {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    analyzer = ComparisonAnalyzer()
    for exp_name, metrics in all_metrics.items():
        analyzer.add_experiment(exp_name, metrics)
    
    # Print comparisons
    print("\nAccuracy Comparison:")
    accuracy_comparison = analyzer.compare_accuracy()
    for exp_name, results in accuracy_comparison.items():
        print(f"  {exp_name:30s}: Final={results['final']:.2f}%, "
              f"Best={results['best']:.2f}%, Mean={results['mean']:.2f}%")
    
    print("\nAggregation Time Comparison:")
    time_comparison = analyzer.compare_time()
    for exp_name, results in time_comparison.items():
        print(f"  {exp_name:30s}: Avg={results['mean']:.4f}s, "
              f"Total={results['total']:.2f}s")
    
    # Plot comparison
    analyzer.plot_comparison()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    
    return all_metrics
