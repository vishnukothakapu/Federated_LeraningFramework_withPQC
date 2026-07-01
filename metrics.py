"""
Evaluation Metrics and Statistics Tracking
"""

import json
import pickle
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts with no display
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path
from config import *

# Resolve paths relative to this file so they work regardless of CWD
_PROJECT_ROOT = Path(__file__).parent
_RESULTS_DIR  = _PROJECT_ROOT / RESULTS_DIR
_LOG_DIR      = _PROJECT_ROOT / LOG_DIR


def _safe_json(obj):
    """Recursively convert numpy types so json.dump never raises TypeError."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


class MetricsTracker:
    """
    Tracks all evaluation metrics for federated learning
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize metrics tracker

        Args:
            experiment_name: Name of experiment
        """
        self.experiment_name = experiment_name
        # Build a filesystem-safe version of the name for use in filenames
        self.safe_name = (
            experiment_name
            .replace(':', '')
            .replace(' ', '_')
            .replace('/', '-')
            .replace('\\', '-')
        )
        self.metrics = {
            'training_loss': [],
            'test_accuracy': [],
            'test_loss': [],
            'attack_success_rate': [],
            'aggregation_time': [],
            'encryption_time': [],
            'decryption_time': [],
            'signature_verification_time': [],
            'communication_overhead': [],
            'num_valid_updates': [],
            'num_invalid_updates': [],
            'krum_rejected_clients': [],
            'client_losses': [],
            'trust_scores': []   # list of {client_id: score} dicts, one per round
        }
        self.round_data = []

        # Ensure output directories exist immediately
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Set up per-experiment log file
        log_file = _LOG_DIR / f"{self.safe_name}.log"
        self._logger = logging.getLogger(self.safe_name)
        self._logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if logger was already created
        if not self._logger.handlers:
            fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S'))
            self._logger.addHandler(fh)

        self._logger.info(f"Experiment started: {experiment_name}")

    def restore(self, metrics_data: dict) -> None:
        """
        Restore metrics from a checkpoint dictionary.

        Called when resuming an experiment after a crash.  Only known keys
        are restored so that adding new metric keys in future does not break
        loading of old checkpoints.

        Args:
            metrics_data: The 'metrics' dict previously saved in progress.json
        """
        for key in self.metrics:
            if key in metrics_data and metrics_data[key]:
                self.metrics[key] = metrics_data[key]
        rounds_restored = len(self.metrics.get('test_accuracy', []))
        self._logger.info(
            f"Metrics restored from checkpoint ({rounds_restored} rounds)"
        )
        print(f"  [Metrics] Restored {rounds_restored} rounds from checkpoint.")

    def save_incremental(self) -> None:
        """
        Flush current metrics to disk immediately (called after every round).

        Writes two files atomically:
          *_metrics_live.json   — full metrics dict (lists of values per round)
          *_summary_live.json   — aggregated summary statistics

        The '_live' suffix distinguishes these in-progress files from the
        final '_metrics.json' / '_summary.json' written at experiment end.
        """
        out = _RESULTS_DIR
        out.mkdir(parents=True, exist_ok=True)

        # Atomic write helper: write to .tmp, then replace
        def _atomic_json(path: Path, data: object) -> None:
            tmp = path.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(_safe_json(data), f, indent=2)
            tmp.replace(path)

        _atomic_json(out / f"{self.safe_name}_metrics_live.json", self.metrics)
        _atomic_json(out / f"{self.safe_name}_summary_live.json",
                     self.get_summary())


    def add_round(self, round_num: int, round_stats: Dict):
        """
        Add metrics from a round.

        Args:
            round_num: Round number
            round_stats: Statistics from that round
        """
        # Training loss
        if 'client_losses' in round_stats and round_stats['client_losses']:
            avg_loss = float(np.mean(round_stats['client_losses']))
            self.metrics['training_loss'].append(avg_loss)
            self.metrics['client_losses'].append(round_stats['client_losses'])

        # Test metrics
        if 'test_accuracy' in round_stats:
            self.metrics['test_accuracy'].append(round_stats['test_accuracy'])
        if 'test_loss' in round_stats:
            self.metrics['test_loss'].append(round_stats['test_loss'])

        # PQC metrics
        pqc_stats = round_stats.get('pqc_stats', {})
        if 'time' in pqc_stats:
            self.metrics['aggregation_time'].append(pqc_stats['time'])
        if 'num_valid' in pqc_stats:
            self.metrics['num_valid_updates'].append(pqc_stats['num_valid'])
        if 'num_invalid' in pqc_stats:
            self.metrics['num_invalid_updates'].append(pqc_stats['num_invalid'])

        # Aggregation/Defense metrics
        agg_stats = round_stats.get('aggregation_stats', {})
        if 'num_rejected' in agg_stats:
            self.metrics['krum_rejected_clients'].append(agg_stats['num_rejected'])

        # Trust scores — store snapshot dict {client_id: score}
        trust_snap = round_stats.get('trust_scores', {})
        if trust_snap:
            self.metrics['trust_scores'].append(
                {str(k): float(v) for k, v in trust_snap.items()}
            )

        # Store round data
        self.round_data.append({
            'round': round_num,
            'stats': round_stats
        })

        # Write a one-line entry to the log file
        acc  = round_stats.get('test_accuracy', 0.0)
        loss = round_stats.get('test_loss', 0.0)
        avg_client_loss = float(np.mean(round_stats['client_losses'])) \
            if round_stats.get('client_losses') else 0.0
        self._logger.info(
            f"Round {round_num + 1:04d} | "
            f"TrainLoss={avg_client_loss:.4f} | "
            f"TestLoss={loss:.4f} | "
            f"Accuracy={acc:.2f}%"
        )
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {}
        
        # Accuracy metrics
        if self.metrics['test_accuracy']:
            summary['final_accuracy'] = self.metrics['test_accuracy'][-1]
            summary['max_accuracy'] = max(self.metrics['test_accuracy'])
            summary['avg_accuracy'] = np.mean(self.metrics['test_accuracy'])
            summary['std_accuracy'] = np.std(self.metrics['test_accuracy'])
        
        # Loss metrics
        if self.metrics['training_loss']:
            summary['final_loss'] = self.metrics['training_loss'][-1]
            summary['min_loss'] = min(self.metrics['training_loss'])
            summary['avg_loss'] = np.mean(self.metrics['training_loss'])
        
        # Timing metrics
        if self.metrics['aggregation_time']:
            summary['avg_aggregation_time'] = np.mean(self.metrics['aggregation_time'])
            summary['total_aggregation_time'] = sum(self.metrics['aggregation_time'])
        
        # PQC metrics
        if self.metrics['encryption_time']:
            summary['avg_encryption_time'] = np.mean(self.metrics['encryption_time'])
        
        if self.metrics['decryption_time']:
            summary['avg_decryption_time'] = np.mean(self.metrics['decryption_time'])
        
        if self.metrics['signature_verification_time']:
            summary['avg_signature_time'] = np.mean(self.metrics['signature_verification_time'])
        
        # Update metrics
        if self.metrics['num_valid_updates']:
            summary['avg_valid_updates'] = np.mean(self.metrics['num_valid_updates'])
        
        if self.metrics['num_invalid_updates']:
            summary['total_invalid_updates'] = sum(self.metrics['num_invalid_updates'])
        
        # Defense metrics (Krum)
        if self.metrics['krum_rejected_clients']:
            summary['avg_krum_rejected_clients'] = np.mean(
                self.metrics['krum_rejected_clients']
            )
        
        return summary
    
    def save_metrics(self, output_dir: str = None):
        """
        Save metrics to the results directory.

        Args:
            output_dir: Override output directory (defaults to RESULTS_DIR)
        """
        out = Path(output_dir) if output_dir else _RESULTS_DIR
        out.mkdir(parents=True, exist_ok=True)

        # --- JSON summary (human-readable) ---
        summary = self.get_summary()
        summary_file = out / f"{self.safe_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(_safe_json(summary), f, indent=2)

        # --- Full metrics as pickle ---
        metrics_file = out / f"{self.safe_name}_metrics.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(self.metrics, f)

        # --- Full metrics as JSON (for easy external loading) ---
        metrics_json_file = out / f"{self.safe_name}_metrics.json"
        with open(metrics_json_file, 'w', encoding='utf-8') as f:
            json.dump(_safe_json(self.metrics), f, indent=2)

        self._logger.info(f"Metrics saved to {out}")
        print(f"Metrics saved to {out}")
    
    def plot_results(self, output_dir: str = None):
        """
        Plot and save figures.

        Args:
            output_dir: Override output directory (defaults to RESULTS_DIR)
        """
        out = Path(output_dir) if output_dir else _RESULTS_DIR
        out.mkdir(parents=True, exist_ok=True)

        # Accuracy plot
        if self.metrics['test_accuracy']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['test_accuracy'], marker='o')
            plt.xlabel('Round')
            plt.ylabel('Test Accuracy (%)')
            plt.title(f'{self.experiment_name} - Test Accuracy')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out / f"{self.safe_name}_accuracy.png", dpi=100)
            plt.close()

        # Loss plot
        if self.metrics['training_loss']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['training_loss'], marker='o')
            plt.xlabel('Round')
            plt.ylabel('Training Loss')
            plt.title(f'{self.experiment_name} - Training Loss')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out / f"{self.safe_name}_loss.png", dpi=100)
            plt.close()

        # Aggregation time plot
        if self.metrics['aggregation_time']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['aggregation_time'], marker='s')
            plt.xlabel('Round')
            plt.ylabel('Time (seconds)')
            plt.title(f'{self.experiment_name} - Aggregation Time')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out / f"{self.safe_name}_agg_time.png", dpi=100)
            plt.close()

        # Valid/invalid updates bar chart — only when both lists are non-empty
        valid   = self.metrics['num_valid_updates']
        invalid = self.metrics['num_invalid_updates']
        if valid and invalid and len(valid) == len(invalid):
            x     = np.arange(len(valid))
            width = 0.35
            plt.figure(figsize=(10, 6))
            plt.bar(x - width / 2, valid,   width, label='Valid')
            plt.bar(x + width / 2, invalid, width, label='Invalid')
            plt.xlabel('Round')
            plt.ylabel('Number of Updates')
            plt.title(f'{self.experiment_name} - Update Verification')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out / f"{self.safe_name}_updates.png", dpi=100)
            plt.close()

        self._logger.info(f"Plots saved to {out}")
        print(f"Plots saved to {out}")

        # Trust score evolution (only when trust scoring was active)
        if self.metrics.get('trust_scores'):
            self._plot_trust_scores(out)

    def _plot_trust_scores(self, out):
        """
        Plot per-client trust score evolution across rounds.

        Only called from plot_results() when trust_scores data exists.
        Each client gets its own line; Byzantine behaviour is visible as
        a downward trend in the corresponding client's score.
        """
        trust_data = self.metrics['trust_scores']  # list of {str(client_id): score}
        if not trust_data:
            return

        # Collect all client ids present across rounds
        all_ids = sorted(
            {int(k) for snap in trust_data for k in snap.keys()},
            key=int
        )

        rounds = list(range(1, len(trust_data) + 1))

        plt.figure(figsize=(12, 6))
        for cid in all_ids:
            scores = [snap.get(str(cid), None) for snap in trust_data]
            # Replace None with NaN for clean gaps in the plot
            scores_clean = [s if s is not None else float('nan') for s in scores]
            plt.plot(rounds, scores_clean, marker='o', markersize=3,
                     label=f'Client {cid}')

        plt.xlabel('Round')
        plt.ylabel('Trust Score')
        plt.title(f'{self.experiment_name} - Client Trust Scores')
        plt.ylim(0.0, 1.05)
        plt.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8,
                    label='Min trust floor')
        plt.legend(loc='lower left', fontsize='small', ncol=2)
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(out / f"{self.safe_name}_trust_scores.png", dpi=100)
        plt.close()
        print(f"Trust score plot saved to {out / (self.safe_name + '_trust_scores.png')}")
    
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:40s}: {value:.4f}")
            else:
                print(f"{key:40s}: {value}")
        print(f"{'='*60}\n")


class ComparisonAnalyzer:
    """Analyze and compare multiple experiments"""
    
    def __init__(self):
        self.experiments = {}
    
    def add_experiment(self, name: str, metrics: Dict):
        """Add experiment metrics"""
        self.experiments[name] = metrics
    
    def compare_accuracy(self):
        """Compare final accuracy across experiments"""
        results = {}
        for name, metrics in self.experiments.items():
            if 'test_accuracy' in metrics and metrics['test_accuracy']:
                results[name] = {
                    'final': metrics['test_accuracy'][-1],
                    'best': max(metrics['test_accuracy']),
                    'mean': np.mean(metrics['test_accuracy'])
                }
        return results
    
    def compare_time(self):
        """Compare aggregation time across experiments"""
        results = {}
        for name, metrics in self.experiments.items():
            if 'aggregation_time' in metrics and metrics['aggregation_time']:
                results[name] = {
                    'mean': np.mean(metrics['aggregation_time']),
                    'total': sum(metrics['aggregation_time']),
                    'std': np.std(metrics['aggregation_time'])
                }
        return results
    
    def plot_comparison(self, output_dir: str = None):
        """Plot comparison of experiments"""
        out = Path(output_dir) if output_dir else _RESULTS_DIR
        out.mkdir(parents=True, exist_ok=True)

        # Accuracy comparison
        plt.figure(figsize=(12, 6))
        for name, metrics in self.experiments.items():
            if 'test_accuracy' in metrics and metrics['test_accuracy']:
                plt.plot(metrics['test_accuracy'], marker='o', label=name)

        plt.xlabel('Round')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Accuracy Comparison Across Experiments')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / "comparison_accuracy.png", dpi=100)
        plt.close()

        # Loss comparison
        plt.figure(figsize=(12, 6))
        for name, metrics in self.experiments.items():
            if 'training_loss' in metrics and metrics['training_loss']:
                plt.plot(metrics['training_loss'], marker='s', label=name)

        plt.xlabel('Round')
        plt.ylabel('Training Loss')
        plt.title('Loss Comparison Across Experiments')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / "comparison_loss.png", dpi=100)
        plt.close()

        print(f"Comparison plots saved to {out}")
