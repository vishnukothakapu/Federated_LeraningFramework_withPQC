"""
Trust-Based Client Scoring for Federated Learning

TrustManager maintains a per-client trust score ∈ [TRUST_MIN, 1.0] that
evolves each round based on:

  1. Cosine similarity   — how well the client's update direction agrees
                           with the (post-aggregation) consensus update.
  2. Norm-ratio penalty  — penalises updates whose L2 norm deviates far
                           from the median norm of all valid updates.
  3. EMA temporal decay  — scores decay smoothly so clients must earn
                           trust continuously (TRUST_ALPHA controls speed).
  4. Sliding window      — a fixed-width history kept for diagnostics /
                           visualisation (governed by TRUST_WINDOW).

Call ordering inside FLServer.aggregate_updates (each round):
    1. weights = trust_mgr.get_weights(valid_client_ids)   # BEFORE aggregation
    2. aggregated = defense.aggregate(updates, trust_weights=weights)
    3. trust_mgr.update_scores(valid_client_ids, updates, aggregated)  # AFTER

This ordering ensures the aggregated update (used as the cosine-similarity
reference) already reflects the trust-weighted consensus, not just a plain
average that would include Byzantine contributions.
"""

import math
import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from config import (
    TRUST_SCORE_ENABLED,
    TRUST_WINDOW,
    TRUST_ALPHA,
    TRUST_MIN,
    TRUST_NORM_PENALTY,
    NUM_CLIENTS,
)


# ---------------------------------------------------------------------------
# Helper: cosine similarity between two 1-D tensors
# ---------------------------------------------------------------------------
def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return cosine similarity in [-1, 1]; safe when either norm is zero."""
    a_flat = a.view(-1).float()
    b_flat = b.view(-1).float()
    norm_a = float(torch.norm(a_flat))
    norm_b = float(torch.norm(b_flat))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# TrustManager
# ---------------------------------------------------------------------------
class TrustManager:
    """
    Manages per-client trust scores for trust-weighted aggregation.

    Attributes
    ----------
    scores : Dict[int, float]
        Current trust score for each client (initialised to 1.0).
    history : Dict[int, deque[float]]
        Sliding window (length TRUST_WINDOW) of per-round scores.
    """

    def __init__(self, client_ids: Optional[List[int]] = None):
        """
        Initialise with equal trust for all clients.

        Args:
            client_ids: List of known client IDs.  If None, IDs are added
                        lazily on the first update.
        """
        ids = client_ids if client_ids is not None else list(range(NUM_CLIENTS))
        self.scores: Dict[int, float] = {cid: 1.0 for cid in ids}
        self.history: Dict[int, deque] = {
            cid: deque(maxlen=TRUST_WINDOW) for cid in ids
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_scores(
        self,
        client_ids: List[int],
        updates: List[torch.Tensor],
        aggregated_update: Optional[torch.Tensor],
    ) -> None:
        """
        Update trust scores for a set of clients after one FL round.

        Called *after* aggregation so the aggregated update is available as
        the consensus reference.

        Args:
            client_ids:        IDs of clients whose updates were accepted.
            updates:           Corresponding update tensors (same order).
            aggregated_update: The aggregated update tensor used as reference.
                               If None, only norm-ratio scoring is applied.
        """
        if not client_ids:
            return

        # ---- Norm statistics for norm-ratio scoring ----
        norms = np.array([
            float(u.norm(2).cpu()) for u in updates
        ], dtype=float)
        median_norm = float(np.median(norms)) if len(norms) > 0 else 1.0

        for cid, update in zip(client_ids, updates):
            self._ensure_client(cid)

            # 1. Cosine-similarity component: maps [-1,1] → [0,1]
            if aggregated_update is not None:
                cos_sim = _cosine_similarity(update, aggregated_update)
                cos_score = (cos_sim + 1.0) / 2.0   # shift to [0, 1]
            else:
                cos_score = 0.5  # neutral when no reference available

            # 2. Norm-ratio penalty: 1.0 if norm ≈ median, decays otherwise
            client_norm = float(update.norm(2).cpu())
            if median_norm > 1e-12:
                ratio = client_norm / median_norm
            else:
                ratio = 1.0
            # Gaussian-shaped penalty centred on ratio=1
            norm_score = math.exp(-((ratio - 1.0) ** 2) / 0.5)

            # 3. Blend cosine and norm contributions
            raw_score = (
                (1.0 - TRUST_NORM_PENALTY) * cos_score
                + TRUST_NORM_PENALTY * norm_score
            )

            # 4. EMA update
            old_score = self.scores[cid]
            new_score = TRUST_ALPHA * old_score + (1.0 - TRUST_ALPHA) * raw_score

            # 5. Clamp to [TRUST_MIN, 1.0]
            new_score = max(TRUST_MIN, min(1.0, new_score))

            self.scores[cid] = new_score
            self.history[cid].append(new_score)

        # Clients that submitted no valid update this round: apply mild decay
        absent = set(self.scores.keys()) - set(client_ids)
        for cid in absent:
            old = self.scores[cid]
            decayed = max(TRUST_MIN, TRUST_ALPHA * old + (1.0 - TRUST_ALPHA) * TRUST_MIN)
            self.scores[cid] = decayed
            self.history[cid].append(decayed)

    def get_weights(self, client_ids: List[int]) -> List[float]:
        """
        Return normalised trust weights for the given client IDs.

        Weights sum to 1.0 so they can be used directly as aggregation
        weights in a weighted average.

        Args:
            client_ids: Ordered list of client IDs.

        Returns:
            List of floats (same order as client_ids), summing to 1.0.
        """
        raw = [self.scores.get(cid, TRUST_MIN) for cid in client_ids]
        total = sum(raw)
        if total < 1e-12:
            # All scores somehow zeroed out — fall back to uniform
            n = len(client_ids)
            return [1.0 / n] * n
        return [w / total for w in raw]

    def get_scores(self) -> Dict[int, float]:
        """Return a snapshot of all current trust scores."""
        return dict(self.scores)

    def get_score(self, client_id: int) -> float:
        """Return the current trust score for a single client."""
        return self.scores.get(client_id, TRUST_MIN)

    def reset(self, client_id: int) -> None:
        """Reset a client's trust score to the minimum (e.g. after misbehaviour)."""
        self._ensure_client(client_id)
        self.scores[client_id] = TRUST_MIN
        self.history[client_id].append(TRUST_MIN)

    def get_krum_modifiers(self, client_ids: List[int]) -> List[float]:
        """
        Return per-client Krum-score multipliers based on trust.

        A lower-trust client gets a *higher* multiplier so its Krum score
        increases and it is less likely to be selected.

        multiplier = 2 - trust_score  ∈ [1, 2-TRUST_MIN]

        Args:
            client_ids: Ordered list of client IDs.

        Returns:
            List of multipliers (same order as client_ids).
        """
        return [2.0 - self.scores.get(cid, TRUST_MIN) for cid in client_ids]

    def get_manhattan_weights(self, client_ids: List[int]) -> np.ndarray:
        """
        Return per-client weight adjustments for Manhattan-distance defense.

        Each client's outlier weight is multiplied by its trust score before
        normalisation, so low-trust clients receive less contribution even
        when their update is not flagged as an outlier.

        Args:
            client_ids: Ordered list of client IDs.

        Returns:
            1-D numpy array of trust scores (same order as client_ids).
        """
        return np.array(
            [self.scores.get(cid, TRUST_MIN) for cid in client_ids],
            dtype=float,
        )

    def print_scores(self, round_num: int) -> None:
        """Pretty-print the current trust scores table."""
        print(f"\n  [Trust Scores] Round {round_num + 1}:")
        print(f"  {'Client':>8} | {'Score':>8} | {'History (last {})'.format(TRUST_WINDOW)}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*20}")
        for cid, score in sorted(self.scores.items()):
            hist = list(self.history[cid])
            hist_str = " ".join(f"{s:.3f}" for s in hist[-5:])
            print(f"  {cid:>8} | {score:>8.4f} | {hist_str}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_client(self, client_id: int) -> None:
        """Lazily register a new client with initial trust = 1.0."""
        if client_id not in self.scores:
            self.scores[client_id] = 1.0
            self.history[client_id] = deque(maxlen=TRUST_WINDOW)
