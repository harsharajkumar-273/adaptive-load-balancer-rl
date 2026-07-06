import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from src.config import (
    NUM_INSTANCES,
    CONTROL_PLANE_INTERVAL_SEC,
    RL_EXPLORATION_PARAM,
    RL_TEMPERATURE,
    RL_REWARD_WEIGHTS,
    CHECKPOINT_PATH
)

class ContextualBanditAgent:
    def __init__(self, num_instances: int, shared_cache):
        self.num_instances = num_instances
        self.shared_cache = shared_cache
        self.active = False
        
        # Context Dimension d = 6
        # Features: [Bias (1.0), CPU/100, Queue/20, P99/200, GlobalRate/150, RateChange/50]
        self.d = 6
        
        # Thompson Sampling parameters for each instance
        # B_i: Precision matrix (d x d)
        self.B = [np.eye(self.d) for _ in range(self.num_instances)]
        # f_i: Reward accumulator vector (d)
        self.f = [np.zeros(self.d) for _ in range(self.num_instances)]
        # theta_hat_i: Mean coefficient vector (d)
        self.theta_hat = [np.zeros(self.d) for _ in range(self.num_instances)]
        
        # Exploration parameter v^2
        self.v2 = RL_EXPLORATION_PARAM
        
        # Track previous context and rate for derivative calculation
        self.prev_context_vectors: Optional[List[np.ndarray]] = None
        self.prev_request_rate = 0.0

        # Load checkpoint if exists
        self.load_checkpoint()

    def save_checkpoint(self, filepath: str = CHECKPOINT_PATH):
        """Saves current precision matrices B and accumulator vectors f to disk."""
        try:
            np.savez(filepath, B=np.array(self.B), f=np.array(self.f))
        except Exception:
            pass

    def load_checkpoint(self, filepath: str = CHECKPOINT_PATH):
        """Loads precision matrices B and accumulator vectors f from disk."""
        try:
            if os.path.exists(filepath):
                data = np.load(filepath)
                B_data = data["B"]
                f_data = data["f"]
                if len(B_data) == self.num_instances and len(f_data) == self.num_instances:
                    self.B = [B_data[i] for i in range(self.num_instances)]
                    self.f = [f_data[i] for i in range(self.num_instances)]
                    # Recompute theta_hat
                    for i in range(self.num_instances):
                        self.theta_hat[i] = np.linalg.solve(self.B[i], self.f[i])
        except Exception:
            pass

    async def start(self):
        self.active = True
        asyncio.create_task(self._run_loop())

    async def stop(self):
        self.active = False

    def _build_context_vector(self, cpu: float, queue: int, p99: float, rate: float, rate_change: float) -> np.ndarray:
        """
        Builds a normalized context vector x_i for an instance.
        """
        return np.array([
            1.0,                       # Bias
            min(1.0, cpu / 100.0),     # CPU utilization (0.0 to 1.0)
            min(1.5, queue / 20.0),    # Queue depth normalized
            min(2.0, p99 / 200.0),     # Recent P99 latency relative to SLA
            min(2.0, rate / 150.0),    # Global request rate
            max(-2.0, min(2.0, rate_change / 50.0))  # Global rate change d/dt
        ], dtype=float)

    def _sample_theta(self, i: int) -> np.ndarray:
        """
        Samples a coefficient vector tilde_theta_i from the posterior distribution:
        N(theta_hat_i, v^2 * B_i^-1)
        """
        # Covariance is B_i^-1
        cov = np.linalg.inv(self.B[i])
        
        # Add small regularization to covariance matrix to ensure positive definiteness
        cov += np.eye(self.d) * 1e-6
        
        # Sample using multivariate normal
        return np.random.multivariate_normal(self.theta_hat[i], self.v2 * cov)

    def select_action_weights(self, context_vectors: List[np.ndarray]) -> List[float]:
        """
        Computes routing weights based on Thompson-sampled expected rewards.
        """
        scores = []
        for i in range(self.num_instances):
            # Sample coefficients from the posterior
            sampled_theta = self._sample_theta(i)
            # Expected reward is the dot product of context and sampled parameters
            score = np.dot(context_vectors[i], sampled_theta)
            scores.append(score)
            
        # Convert expected rewards (negative values) to probability weights using Softmax
        scores = np.array(scores, dtype=float)
        
        # Subtract max score for numerical stability under exponentiation
        stable_scores = (scores - np.max(scores)) / RL_TEMPERATURE
        exp_scores = np.exp(stable_scores)
        weights = exp_scores / np.sum(exp_scores)
        
        return weights.tolist()

    def update_model(self, feedback_data: Dict[str, Any]):
        """
        Performs Bayesian updates on LinTS parameters using observed window feedback.
        """
        if self.prev_context_vectors is None:
            return  # Skip update on first iteration due to lack of historical context

        feedback_list = feedback_data["feedback"]
        
        for feed in feedback_list:
            idx = feed["instance_idx"]
            reqs = feed["requests_count"]
            
            # Update only if this backend received traffic (bandit active feedback)
            if reqs > 0:
                p99 = feed["p99"]
                err_rate = feed["error_rate"]
                sla_breach = feed["sla_breach_rate"]
                
                # Calculate penalty-driven reward (higher values are better, i.e. closer to 0)
                latency_penalty = p99 / 200.0
                reward = -(
                    RL_REWARD_WEIGHTS["latency"] * latency_penalty +
                    RL_REWARD_WEIGHTS["error"] * err_rate +
                    RL_REWARD_WEIGHTS["sla"] * sla_breach
                )
                
                # Clip reward to avoid extreme updates from outliers
                reward = max(-15.0, reward)
                
                # Get the context vector that was active when decisions were made
                x = self.prev_context_vectors[idx]
                
                # B_i <- B_i + x * x^T
                self.B[idx] += np.outer(x, x)
                # f_i <- f_i + r * x
                self.f[idx] += reward * x
                # theta_hat_i <- B_i^-1 * f_i
                self.theta_hat[idx] = np.linalg.solve(self.B[idx], self.f[idx])

    async def _run_loop(self):
        """Main asynchronous Control Plane loop."""
        # Warmup delay
        await asyncio.sleep(0.5)
        
        while self.active:
            try:
                # 1. Fetch current metrics and calculate derivative context
                metrics = self.shared_cache.get_instance_metrics()
                telemetry = self.shared_cache.get_telemetry()
                
                curr_rate = telemetry["global_request_rate"]
                rate_change = (curr_rate - self.prev_request_rate) / CONTROL_PLANE_INTERVAL_SEC
                self.prev_request_rate = curr_rate
                
                # 2. Build context vectors for all instances
                context_vectors = []
                for i in range(self.num_instances):
                    cpu = metrics["cpu"][i]
                    queue = metrics["queue"][i]
                    p99 = metrics["p99"][i]
                    
                    x_i = self._build_context_vector(cpu, queue, p99, curr_rate, rate_change)
                    context_vectors.append(x_i)
                
                # 3. Flush previous window feedback & Update bandit model
                # This gathers what actually happened in the last 150ms
                window_data = self.shared_cache.flush_window_feedback()
                self.update_model(window_data)
                
                # 4. Generate new routing weights
                new_weights = self.select_action_weights(context_vectors)
                
                # 5. Push to Shared Memory Cache
                self.shared_cache.set_routing_weights(new_weights)
                
                # Save context vectors for feedback alignment in the next step
                self.prev_context_vectors = context_vectors
                
            except Exception as e:
                # Prevent control loop crash
                pass
                
            # Sleep for the control plane interval
            await asyncio.sleep(CONTROL_PLANE_INTERVAL_SEC)
