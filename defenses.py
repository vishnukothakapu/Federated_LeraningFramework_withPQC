"""
Defense Mechanisms: FedAvg, Krum, and ManhattanDistance
All three defenses optionally accept trust weights from a TrustManager.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from config import *


class FedAvgDefense:
    """
    Standard FedAvg aggregation (baseline defense)
    Averages all updates; accepts optional trust weights.
    """
    
    def __init__(self):
        self.name = 'FedAvg'
    
    def aggregate(self, updates: List[torch.Tensor], 
                 weights: List[float] = None,
                 trust_weights: List[float] = None) -> torch.Tensor:
        """
        Aggregate updates using FedAvg.

        When trust_weights are provided they are multiplied element-wise with
        any explicit weights before normalisation, so the result is a single
        trust-adjusted weighted average.
        
        Args:
            updates:       List of model updates
            weights:       Optional base weights for each update
            trust_weights: Optional per-client trust weights from TrustManager
        
        Returns:
            Aggregated update
        """
        if not updates:
            return None
        
        n = len(updates)

        # Start with uniform base weights if none supplied
        if weights is None:
            weights = [1.0 / n] * n

        # Blend in trust weights when provided
        if trust_weights is not None and len(trust_weights) == n:
            combined = [w * t for w, t in zip(weights, trust_weights)]
            total = sum(combined)
            if total > 1e-12:
                weights = [c / total for c in combined]
            # else fall back to original weights unchanged
        
        # Weighted average
        aggregated = None
        for update, weight in zip(updates, weights):
            if aggregated is None:
                aggregated = weight * update
            else:
                aggregated += weight * update
        
        return aggregated


class KrumDefense:
    """
    Krum Defense: Byzantine-resilient aggregation (Blanchard et al., 2017)

    Algorithm (Multi-Krum):
    1. For each update i, compute its score = sum of squared L2 distances to
       its (n - f - 2) nearest neighbours (excluding itself).
    2. Select the m updates with the lowest Krum scores.
    3. Return the average of those m selected updates.

    With m == 1 this is standard (single) Krum;
    with m > 1 it is Multi-Krum, which has better accuracy at a slight
    reduction in Byzantine-resilience.
    """

    def __init__(self,
                 num_byzantine: int = KRUM_NUM_BYZANTINE,
                 multi_k: int = KRUM_MULTI_K):
        """
        Initialize Krum defense.

        Args:
            num_byzantine: Expected maximum number of Byzantine clients (f).
            multi_k:       Number of updates to select for Multi-Krum (m).
                           Set to 1 for standard (single) Krum.
        """
        self.name = 'Krum'
        self.num_byzantine = num_byzantine
        self.multi_k = multi_k

    def _squared_l2_distance(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Compute squared Euclidean distance between two flattened tensors."""
        diff = v1.view(-1) - v2.view(-1)
        return float(torch.sum(diff ** 2).cpu().item())

    def compute_krum_scores(self, updates: List[torch.Tensor]) -> np.ndarray:
        """
        Compute Krum score for each update.

        The score for update i is the sum of squared distances to its
        (n - f - 2) nearest neighbours.

        Args:
            updates: List of model updates (length n).

        Returns:
            scores: 1-D numpy array of shape (n,).
        """
        n = len(updates)
        f = self.num_byzantine
        # Number of neighbours to consider
        num_neighbours = max(n - f - 2, 1)

        # Build full pairwise distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._squared_l2_distance(updates[i], updates[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Krum score = sum of distances to the num_neighbours closest peers
        scores = np.zeros(n)
        for i in range(n):
            # Distances from i to all others (exclude self)
            dists = np.concatenate([dist_matrix[i, :i], dist_matrix[i, i+1:]])
            # Sort and take the smallest num_neighbours
            scores[i] = np.sum(np.sort(dists)[:num_neighbours])

        return scores

    def aggregate(self, updates: List[torch.Tensor],
                  client_ids: List[int] = None,
                  trust_modifiers: List[float] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate updates with Multi-Krum defense.

        When trust_modifiers are provided (from TrustManager.get_krum_modifiers),
        each client's Krum score is multiplied by its modifier so lower-trust
        clients rank worse and are less likely to be selected.

        Args:
            updates:          List of model updates.
            client_ids:       Optional list of client IDs (for logging).
            trust_modifiers:  Optional per-client multipliers from TrustManager.

        Returns:
            (aggregated_update, defense_info)
        """
        if not updates:
            return None, {}

        n = len(updates)
        # Multi-Krum selects at most (n - num_byzantine) updates
        m = min(self.multi_k, n - self.num_byzantine)
        m = max(m, 1)  # Always select at least one

        scores = self.compute_krum_scores(updates)

        # Apply trust modifiers: higher modifier → higher score → less likely selected
        if trust_modifiers is not None and len(trust_modifiers) == n:
            scores = scores * np.array(trust_modifiers, dtype=float)

        # Select the m updates with lowest Krum scores
        selected_indices = np.argsort(scores)[:m]

        # Average selected updates
        aggregated = None
        for idx in selected_indices:
            if aggregated is None:
                aggregated = updates[idx].clone()
            else:
                aggregated += updates[idx]
        aggregated = aggregated / m

        selected_ids = (
            [client_ids[i] for i in selected_indices]
            if client_ids is not None else selected_indices.tolist()
        )
        rejected_ids = (
            [client_ids[i] for i in range(n) if i not in selected_indices]
            if client_ids is not None
            else [i for i in range(n) if i not in selected_indices]
        )

        defense_info = {
            'krum_scores':       [float(s) for s in scores],
            'selected_indices':  selected_indices.tolist(),
            'selected_ids':      [int(x) for x in selected_ids],
            'rejected_ids':      [int(x) for x in rejected_ids],
            'num_selected':      int(m),
            'num_rejected':      int(n - m),
        }

        return aggregated, defense_info


class ManhattanDistanceDefense:
    """
    Manhattan Distance Defense: Detects anomalous updates based on L1 distance
    
    Algorithm:
    1. Compute Manhattan distance from each update to the median
    2. Identify outliers (updates far from median)
    3. Reduce weight of outlier updates
    4. Aggregate weighted updates
    """
    
    def __init__(self, threshold: float = MANHATTAN_DISTANCE_THRESHOLD,
                 deviation_factor: float = MANHATTAN_DISTANCE_DEVIATION_FACTOR):
        """
        Initialize Manhattan Distance defense
        
        Args:
            threshold: Distance threshold multiplier for deviation
            deviation_factor: Multiplier for standard deviation in outlier detection
        """
        self.name = 'ManhattanDistance'
        self.threshold = threshold
        self.deviation_factor = deviation_factor
    
    def _compute_manhattan_distance(self, v1: torch.Tensor, 
                                   v2: torch.Tensor) -> float:
        """Compute Manhattan distance (L1 norm) between two vectors"""
        # Flatten vectors
        v1_flat = v1.view(-1)
        v2_flat = v2.view(-1)
        
        # Compute L1 distance
        distance = torch.sum(torch.abs(v1_flat - v2_flat))
        return float(distance.cpu().detach().numpy())
    
    def compute_median_update(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute element-wise median of all updates
        
        Args:
            updates: List of model updates
        
        Returns:
            Median update tensor
        """
        # Stack all updates
        stacked = torch.stack([u.view(-1) for u in updates], dim=1)
        
        # Compute median along dimension 1 (across clients)
        median = torch.median(stacked, dim=1)[0]
        
        # Return in original shape (use first update's shape)
        return median.view(updates[0].shape)
    
    def detect_outliers(self, updates: List[torch.Tensor],
                       median_update: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outlier updates based on Manhattan distance
        
        Args:
            updates: List of model updates
            median_update: Median update
        
        Returns:
            (distances, outlier_indices)
        """
        num_updates = len(updates)
        distances = np.zeros(num_updates)
        
        # Compute distance from each update to median
        for i, update in enumerate(updates):
            distances[i] = self._compute_manhattan_distance(update, median_update)
        
        # Compute mean and standard deviation of distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Identify outliers: distance > mean + deviation_factor * std
        threshold_distance = mean_distance + self.deviation_factor * std_distance
        outlier_indices = np.where(distances > threshold_distance)[0]
        
        return distances, outlier_indices
    
    def aggregate(self, updates: List[torch.Tensor],
                 client_ids: List[int] = None,
                 trust_weights: np.ndarray = None) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate updates with Manhattan Distance defense.

        When trust_weights are provided (from TrustManager.get_manhattan_weights),
        each client's outlier weight is multiplied by its trust score before
        normalisation, further down-weighting low-trust clients.
        
        Args:
            updates:       List of model updates
            client_ids:    Client IDs (optional)
            trust_weights: Optional per-client trust multipliers (numpy array)
        
        Returns:
            (aggregated_update, defense_info)
        """
        if not updates:
            return None, {}
        
        # Compute median update
        median_update = self.compute_median_update(updates)
        
        # Detect outliers
        distances, outlier_indices = self.detect_outliers(updates, median_update)
        
        # Compute weights for aggregation
        weights = np.ones(len(updates))
        
        # Reduce weights for outlier updates
        for idx in outlier_indices:
            weights[idx] *= 0.5  # Reduce contribution

        # Apply trust multipliers before normalisation
        if trust_weights is not None and len(trust_weights) == len(updates):
            weights = weights * trust_weights
        
        # Normalize weights
        total_w = np.sum(weights)
        if total_w > 1e-12:
            weights = weights / total_w
        else:
            weights = np.ones(len(updates)) / len(updates)
        
        # Aggregate with weighted average
        aggregated = None
        for i, (update, weight) in enumerate(zip(updates, weights)):
            if aggregated is None:
                aggregated = weight * update
            else:
                aggregated += weight * update
        
        # Store stats — only JSON-serialisable scalars/lists
        defense_info = {
            'distances':      [float(d) for d in distances],
            'outlier_indices': outlier_indices.tolist(),
            'weights':        [float(w) for w in weights],
            'num_outliers':   int(len(outlier_indices)),
            'mean_distance':  float(np.mean(distances)),
            'std_distance':   float(np.std(distances)),
            'num_rejected':   int(len(outlier_indices)),
        }

        return aggregated, defense_info


class AdaptiveDefense:
    """Base class for adaptive defenses that can switch strategies"""

    def __init__(self):
        self.defense_methods = {
            'fedavg': FedAvgDefense(),
            'krum': KrumDefense(),
            'manhattan': ManhattanDistanceDefense()
        }
        self.current_defense = 'fedavg'

    def set_defense(self, defense_name: str):
        """Set current defense method"""
        if defense_name in self.defense_methods:
            self.current_defense = defense_name
        else:
            raise ValueError(f"Unknown defense: {defense_name}")

    def aggregate(self, updates: List[torch.Tensor],
                  client_ids: List[int] = None,
                  trust_weights=None,
                  trust_modifiers=None) -> Tuple[torch.Tensor, Dict]:
        """Aggregate using current defense method, forwarding trust arguments."""
        defense = self.defense_methods[self.current_defense]

        if self.current_defense == 'fedavg':
            return defense.aggregate(updates, trust_weights=trust_weights), {'defense': 'FedAvg'}
        elif self.current_defense == 'krum':
            return defense.aggregate(updates, client_ids, trust_modifiers=trust_modifiers)
        elif self.current_defense == 'manhattan':
            return defense.aggregate(updates, client_ids, trust_weights=trust_weights)


def create_defense(defense_name: str = DEFENSE_METHOD,
                   trust_manager=None) -> object:
    """
    Factory function to create defense mechanism.

    Args:
        defense_name:  'fedavg', 'krum', or 'manhattan'
        trust_manager: Optional TrustManager instance; stored on the defense
                       object as ``.trust_manager`` for use during aggregation.

    Returns:
        Defense instance
    """
    if defense_name == 'fedavg':
        defense = FedAvgDefense()
    elif defense_name == 'krum':
        defense = KrumDefense()
    elif defense_name == 'manhattan':
        defense = ManhattanDistanceDefense()
    else:
        raise ValueError(f"Unknown defense: {defense_name}")

    # Attach trust manager as optional attribute for use in FLServer
    defense.trust_manager = trust_manager
    return defense
