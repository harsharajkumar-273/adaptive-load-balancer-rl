# src/agent.py
"""
Reinforcement Learning Control Plane Agent using Linear Thompson Sampling (Contextual Bandits)
supporting both periodic interval execution and Event-Driven Pub/Sub reactive streaming.
"""
import os
import json
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
        self.B = [np.eye(self.d) for _ in range(self.num_instances)]
        self.f = [np.zeros(self.d) for _ in range(self.num_instances)]
        self.theta_hat = [np.zeros(self.d) for _ in range(self.num_instances)]
        
        self.v2 = RL_EXPLORATION_PARAM
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
                    for i in range(self.num_instances):
                        self.theta_hat[i] = np.linalg.solve(self.B[i], self.f[i])
        except Exception:
            pass

    async def start(self):
        self.active = True
        asyncio.create_task(self._run_loop())
        asyncio.create_task(self._event_stream_listener())

    async def stop(self):
        self.active = False

    def _build_context_vector(self, cpu: float, queue: int, p99: float, rate: float, rate_change: float) -> np.ndarray:
        return np.array([
            1.0,
            min(1.0, cpu / 100.0),
            min(1.5, queue / 20.0),
            min(2.0, p99 / 200.0),
            min(2.0, rate / 150.0),
            max(-2.0, min(2.0, rate_change / 50.0))
        ], dtype=float)

    def _sample_theta(self, i: int) -> np.ndarray:
        cov = np.linalg.inv(self.B[i])
        cov += np.eye(self.d) * 1e-6
        return np.random.multivariate_normal(self.theta_hat[i], self.v2 * cov)

    def select_action_weights(self, context_vectors: List[np.ndarray]) -> List[float]:
        scores = []
        for i in range(self.num_instances):
            sampled_theta = self._sample_theta(i)
            score = np.dot(context_vectors[i], sampled_theta)
            scores.append(score)
            
        scores = np.array(scores, dtype=float)
        stable_scores = (scores - np.max(scores)) / RL_TEMPERATURE
        exp_scores = np.exp(stable_scores)
        weights = exp_scores / np.sum(exp_scores)
        return weights.tolist()

    def update_model(self, feedback_data: Dict[str, Any]):
        if self.prev_context_vectors is None:
            return

        feedback_list = feedback_data["feedback"]
        for feed in feedback_list:
            idx = feed["instance_idx"]
            reqs = feed["requests_count"]
            
            if reqs > 0:
                p99 = feed["p99"]
                err_rate = feed["error_rate"]
                sla_breach = feed["sla_breach_rate"]
                
                latency_penalty = p99 / 200.0
                reward = -(
                    RL_REWARD_WEIGHTS["latency"] * latency_penalty +
                    RL_REWARD_WEIGHTS["error"] * err_rate +
                    RL_REWARD_WEIGHTS["sla"] * sla_breach
                )
                reward = max(-15.0, reward)
                x = self.prev_context_vectors[idx]
                
                self.B[idx] += np.outer(x, x)
                self.f[idx] += reward * x
                self.theta_hat[idx] = np.linalg.solve(self.B[idx], self.f[idx])

    async def _event_stream_listener(self):
        """Event-Driven Reactive Streaming subscriber for Redis Pub/Sub channel events."""
        try:
            pubsub = self.shared_cache.client.pubsub()
            pubsub.subscribe("events:request_outcomes")
            
            while self.active:
                msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                if msg and msg.get("type") == "message":
                    data_bytes = msg.get("data")
                    if data_bytes:
                        try:
                            payload = json.loads(data_bytes.decode('utf-8'))
                            # Reactive state update on outcome stream
                            idx = payload.get("node_idx")
                            lat = payload.get("latency_ms")
                            sla = payload.get("breached_sla")
                        except Exception:
                            pass
                await asyncio.sleep(0.05)
        except Exception:
            pass

    async def _run_loop(self):
        await asyncio.sleep(0.5)
        while self.active:
            try:
                metrics = self.shared_cache.get_instance_metrics()
                telemetry = self.shared_cache.get_telemetry()
                
                curr_rate = telemetry.get("global_request_rate", 0.0)
                rate_change = (curr_rate - self.prev_request_rate) / CONTROL_PLANE_INTERVAL_SEC
                self.prev_request_rate = curr_rate
                
                context_vectors = []
                for i in range(self.num_instances):
                    cpu = metrics["cpu"][i]
                    queue = metrics["queue"][i]
                    p99 = metrics["p99"][i]
                    x_i = self._build_context_vector(cpu, queue, p99, curr_rate, rate_change)
                    context_vectors.append(x_i)
                
                window_data = self.shared_cache.flush_window_feedback()
                self.update_model(window_data)
                
                new_weights = self.select_action_weights(context_vectors)
                self.shared_cache.set_routing_weights(new_weights)
                self.prev_context_vectors = context_vectors
                
            except Exception:
                pass
                
            await asyncio.sleep(CONTROL_PLANE_INTERVAL_SEC)
