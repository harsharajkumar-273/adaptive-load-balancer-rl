# src/routing_strategies.py
"""
Multi-strategy routing engine offering 5 baseline & adaptive load balancing algorithms:
1. lin_ts: Reinforcement Learning Linear Thompson Sampling Contextual Bandit
2. least_conn: Dynamic Least Connections
3. p2c: Power of Two Choices
4. round_robin: Sequential Round Robin
5. weighted_round_robin: Capacity-Weighted Round Robin

The fleet size is taken from the metrics passed to each call, so nodes added or
removed by the autoscaler are routable immediately.
"""
import random
import threading
from typing import List, Tuple, Optional
from src.config import BACKEND_SPECS
from src.learned_routing import LEARNED_STRATEGIES, LearnedRouter

# Nodes whose CPU exceeds this are masked out (weight 0) when masking is enabled.
CPU_MASK_THRESHOLD = 85.0
# Capacity assumed for autoscaled nodes that have no entry in BACKEND_SPECS
# (matches the spec generated in backend_node.py).
DEFAULT_AUTOSCALED_CAPACITY = 100.0

VALID_STRATEGIES = ["lin_ts", "least_conn", "p2c", "round_robin", "weighted_round_robin"] + LEARNED_STRATEGIES


def node_capacity(idx: int) -> float:
    if idx < len(BACKEND_SPECS):
        return BACKEND_SPECS[idx].get("capacity", 50.0)
    return DEFAULT_AUTOSCALED_CAPACITY


class RoutingEngine:
    def __init__(self, num_instances: Optional[int] = None, apply_cpu_mask: bool = True,
                 learned: Optional[LearnedRouter] = None):
        # num_instances is kept for backwards compatibility; the effective fleet
        # size is always len(cpu_list) at call time.
        self.num_instances = num_instances
        self.apply_cpu_mask = apply_cpu_mask
        self.learned = learned or LearnedRouter()
        self._rr_counter = 0
        self._lock = threading.Lock()

    def capacity_weights(self, n: int) -> List[float]:
        capacities = [node_capacity(i) for i in range(n)]
        total_cap = sum(capacities)
        return [c / total_cap for c in capacities] if total_cap > 0 else [1.0 / n] * n

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
        n = len(cpu_list)
        self.num_instances = n

        # 1. Action Masking Guardrail: override weights to 0 if CPU is too high.
        #    Applied identically to every strategy so comparisons are fair.
        masked_weights = []
        for idx in range(n):
            if self.apply_cpu_mask and cpu_list[idx] > CPU_MASK_THRESHOLD:
                masked_weights.append(0.0)
            else:
                masked_weights.append(weights[idx] if idx < len(weights) else 1.0 / n)

        total_masked_weight = sum(masked_weights)
        if total_masked_weight > 0:
            effective_weights = [w / total_masked_weight for w in masked_weights]
        else:
            # Emergency: all nodes overloaded -> least connections over the whole fleet
            return self._select_least_connections(queue_list), "fallback_all_masked_least_conn", [1.0 / n] * n

        # Strategy Dispatcher
        if strategy == "lin_ts":
            chosen_idx = random.choices(range(n), weights=effective_weights, k=1)[0]
            return chosen_idx, "rl_adaptive_lin_ts", effective_weights

        elif strategy == "least_conn":
            chosen_idx = self._select_least_connections(queue_list, masked_weights=masked_weights)
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

        elif strategy in LEARNED_STRATEGIES:
            eligible = [i for i, w in enumerate(masked_weights) if w > 0.0]
            chosen_idx = self.learned.select(strategy, queue_list, eligible)
            return chosen_idx, strategy, effective_weights

        else:
            # Default fallback to LinTS
            chosen_idx = random.choices(range(n), weights=effective_weights, k=1)[0]
            return chosen_idx, "rl_adaptive_lin_ts", effective_weights

    def _select_least_connections(self, queue_list: List[int], masked_weights: List[float] = None) -> int:
        """Selects backend with lowest active queue depth among unmasked nodes."""
        valid_indices = [
            idx for idx in range(len(queue_list))
            if not masked_weights or masked_weights[idx] > 0.0
        ]
        if not valid_indices:
            valid_indices = list(range(len(queue_list)))

        min_q = min(queue_list[i] for i in valid_indices)
        best_candidates = [i for i in valid_indices if queue_list[i] == min_q]
        return random.choice(best_candidates)

    def _select_power_of_two_choices(self, queue_list: List[int], masked_weights: List[float]) -> int:
        """P2C algorithm: Samples two random unmasked nodes and picks the one with fewer connections."""
        valid_indices = [i for i, w in enumerate(masked_weights) if w > 0.0]
        if not valid_indices:
            valid_indices = list(range(len(masked_weights)))

        if len(valid_indices) == 1:
            return valid_indices[0]

        c1, c2 = random.sample(valid_indices, 2)
        return c1 if queue_list[c1] <= queue_list[c2] else c2

    def _select_round_robin(self, masked_weights: List[float]) -> int:
        """Sequential round robin across available nodes."""
        n = len(masked_weights)
        with self._lock:
            for _ in range(n):
                idx = self._rr_counter % n
                self._rr_counter += 1
                if masked_weights[idx] > 0.0:
                    return idx
            # If all masked
            idx = self._rr_counter % n
            self._rr_counter += 1
            return idx

    def _select_weighted_round_robin(self, masked_weights: List[float]) -> int:
        """Stochastic selection using static capacity weights combined with action masks."""
        n = len(masked_weights)
        cap = self.capacity_weights(n)
        combined_weights = [cap[i] if masked_weights[i] > 0.0 else 0.0 for i in range(n)]
        if sum(combined_weights) > 0:
            return random.choices(range(n), weights=combined_weights, k=1)[0]
        return self._select_round_robin(masked_weights)
