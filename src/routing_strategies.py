# src/routing_strategies.py
"""
Multi-strategy routing engine offering 5 baseline & adaptive load balancing algorithms:
1. lin_ts: Reinforcement Learning Linear Thompson Sampling Contextual Bandit
2. least_conn: Dynamic Least Connections
3. p2c: Power of Two Choices
4. round_robin: Sequential Round Robin
5. weighted_round_robin: Capacity-Weighted Round Robin
"""
import random
import threading
from typing import List, Tuple, Dict, Any
from src.config import BACKEND_SPECS

class RoutingEngine:
    def __init__(self, num_instances: int):
        self.num_instances = num_instances
        self._rr_counter = 0
        self._lock = threading.Lock()
        
        # Calculate weights for weighted round robin based on capacity specs
        capacities = [spec.get("capacity", 50.0) for spec in BACKEND_SPECS[:num_instances]]
        total_cap = sum(capacities)
        self.capacity_weights = [c / total_cap for c in capacities] if total_cap > 0 else [1.0 / num_instances] * num_instances

    def select_instance(
        self,
        strategy: str,
        weights: List[float],
        cpu_list: List[float],
        queue_list: List[int]
    ) -> Tuple[int, str, List[float]]:
        """
        Selects a target backend node index based on the chosen strategy.
        Returns: (chosen_idx, routing_mode_name, effective_weights)
        """
        strategy = strategy.lower()

        # 1. Action Masking Guardrail: override weights to 0 if CPU > 85%
        masked_weights = []
        for idx in range(self.num_instances):
            if cpu_list[idx] > 85.0:
                masked_weights.append(0.0)
            else:
                masked_weights.append(weights[idx] if idx < len(weights) else 1.0 / self.num_instances)

        total_masked_weight = sum(masked_weights)
        if total_masked_weight > 0:
            effective_weights = [w / total_masked_weight for w in masked_weights]
        else:
            # Emergency: All nodes overloaded -> Fallback to least connections among healthy
            return self._select_least_connections(queue_list, masked=True), "fallback_all_masked_least_conn", [0.2] * self.num_instances

        # Strategy Dispatcher
        if strategy == "lin_ts":
            chosen_idx = random.choices(range(self.num_instances), weights=effective_weights, k=1)[0]
            return chosen_idx, "rl_adaptive_lin_ts", effective_weights

        elif strategy == "least_conn":
            chosen_idx = self._select_least_connections(queue_list, masked=False, masked_weights=masked_weights)
            return chosen_idx, "baseline_least_connections", effective_weights

        elif strategy == "p2c":
            chosen_idx = self._select_power_of_two_choices(queue_list, masked_weights)
            return chosen_idx, "baseline_power_of_two_choices", effective_weights

        elif strategy == "round_robin":
            chosen_idx = self._select_round_robin(masked_weights)
            return chosen_idx, "baseline_round_robin", effective_weights

        elif strategy == "weighted_round_robin":
            chosen_idx = self._select_weighted_round_robin(masked_weights)
            return chosen_idx, "baseline_weighted_round_robin", effective_weights

        else:
            # Default fallback to LinTS
            chosen_idx = random.choices(range(self.num_instances), weights=effective_weights, k=1)[0]
            return chosen_idx, "rl_adaptive_lin_ts", effective_weights

    def _select_least_connections(self, queue_list: List[int], masked: bool = False, masked_weights: List[float] = None) -> int:
        """Selects backend with lowest active queue depth."""
        valid_indices = []
        for idx, q in enumerate(queue_list):
            if not masked and masked_weights and masked_weights[idx] == 0.0:
                continue
            valid_indices.append(idx)
            
        if not valid_indices:
            valid_indices = list(range(self.num_instances))

        min_q = min(queue_list[i] for i in valid_indices)
        best_candidates = [i for i in valid_indices if queue_list[i] == min_q]
        return random.choice(best_candidates)

    def _select_power_of_two_choices(self, queue_list: List[int], masked_weights: List[float]) -> int:
        """P2C algorithm: Samples two random unmasked nodes and picks the one with fewer connections."""
        valid_indices = [i for i, w in enumerate(masked_weights) if w > 0.0]
        if not valid_indices:
            valid_indices = list(range(self.num_instances))
            
        if len(valid_indices) == 1:
            return valid_indices[0]
            
        # Sample 2 distinct candidates
        c1, c2 = random.sample(valid_indices, 2)
        if queue_list[c1] <= queue_list[c2]:
            return c1
        else:
            return c2

    def _select_round_robin(self, masked_weights: List[float]) -> int:
        """Sequential round robin across available nodes."""
        with self._lock:
            for _ in range(self.num_instances):
                idx = self._rr_counter % self.num_instances
                self._rr_counter += 1
                if masked_weights[idx] > 0.0:
                    return idx
            # If all masked
            idx = self._rr_counter % self.num_instances
            self._rr_counter += 1
            return idx

    def _select_weighted_round_robin(self, masked_weights: List[float]) -> int:
        """Stochastic selection using static capacity weights combined with action masks."""
        combined_weights = [self.capacity_weights[i] if masked_weights[i] > 0.0 else 0.0 for i in range(self.num_instances)]
        total = sum(combined_weights)
        if total > 0:
            normalized = [w / total for w in combined_weights]
            return random.choices(range(self.num_instances), weights=normalized, k=1)[0]
        return self._select_round_robin(masked_weights)
