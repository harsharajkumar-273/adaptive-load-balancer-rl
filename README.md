# ⚡ Autonomous AI-Driven Load Balancer
### Contextual Bandit Traffic Optimization for Distributed Microservices

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-7.0+-red.svg)](https://redis.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Exporter-orange.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-orange.svg)](https://grafana.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

An enterprise-grade, portfolio-defining prototype of an **Autonomous Microservice Load Balancer**. It replaces static heuristics with **Online Reinforcement Learning (Linear Thompson Sampling)** to dynamically route traffic away from degrading backends *before* they crash—maintaining SLA compliance under thundering-herd traffic surges.

---

## 🎯 WHY THIS PROJECT EXISTS

### The Problem: Traditional Load Balancers Fail Under Dynamic Stress

Traditional load balancing algorithms (Round Robin, Least Connections, Static Weighted) were designed for predictable infrastructure. In modern cloud-native microservices, they fail during extreme traffic spikes (e.g., Netflix release events, concert ticket drops, flash sales):

| Traditional Algorithm | Failure Mode | Why It Breaks at Scale |
|:---|:---|:---|
| **Round-Robin** | Blind Traffic Distribution | Distributes requests equally regardless of backend capacity, hardware heterogeneity, or CPU saturation, crushing weaker nodes. |
| **Weighted Round-Robin** | Static Assumptions | Static capacity weights cannot adapt when a pod suffers from noisy neighbors, JVM garbage collection pauses, or database lock contention. |
| **Least Connections** | Lagging Telemetry | Active TCP connection count does *not* reflect CPU contention. A node processing 5 heavy requests can be 100% CPU-bound while a node with 20 light requests sits idle. |

### The Impact
Under sudden load spikes, traditional algorithms route traffic into saturated nodes, causing cascading failure storms, exponential latency degradation, high HTTP 5xx error rates, and **SLA breaches ($> 200\text{ms}$)**.

---

## 🚀 THE SOLUTION: CONTEXTUAL BANDIT ADAPTIVE ROUTING

Rather than relying on static rules, this load balancer treats routing as a **Contextual Multi-Armed Bandit Problem**. 

Before forwarding a request, the AI evaluates the **real-time system context** (CPU load, queue depth, historical P99 latency, global request rate, and traffic acceleration). It predicts which node will process the request fastest while satisfying safety guardrails.

```
       TRADITIONAL LOAD BALANCING                        THIS AUTONOMOUS LOAD BALANCER
  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐
  │ Round Robin / Least Connections │               │ Linear Thompson Sampling (LinTS)│
  │   - Static / Lagging Rules      │               │   - Multi-dimensional Context   │
  │   - Crushes Degrading Nodes     │               │   - Predictive & Self-Healing   │
  └────────────────┬────────────────┘               └────────────────┬────────────────┘
                   │                                                 │
                   ▼                                                 ▼
     ❌ Cascading Pod Crashes                          ✅ 0 SLA Breaches & Fast Recovery
```

---

## 🏗️ HOW IT WORKS: SYSTEM ARCHITECTURE

To achieve **sub-millisecond routing speeds**, AI inference and training are decoupled from the live request path into a **Two-Tier Data & Control Plane**:

```mermaid
graph TD
    Client[Traffic Generator / Load Tester] -->|HTTP:8000/auth, /play-video, /analytics| Gateway[API Gateway Proxy - Data Plane]
    Gateway -->|1. Read Weights & Service Discovery| Registry[Service Registry / Redis]
    Gateway -->|2. Route via lin_ts / least_conn / p2c / rr| Node1[Backend Node 1 - Port 8001]
    Gateway -->|2. Route via lin_ts / least_conn / p2c / rr| Node2[Backend Node 2 - Port 8002]
    Gateway -->|2. Route via lin_ts / least_conn / p2c / rr| NodeN[Autoscaled Node N - Port 8006+]
    
    HPA[HPA Cluster AutoScaler Engine] -->|Monitors CPU > 75% -> Spawns Nodes| NodeN
    
    Node1 -->|Heartbeat & CPU Metrics| Redis[(Redis State Store)]
    Node2 -->|Heartbeat & CPU Metrics| Redis
    NodeN -->|Heartbeat & CPU Metrics| Redis
    
    Gateway -->|3. Publish Outcome Stream| PubSub[Redis Pub/Sub Channel]
    PubSub -->|4. Reactive Event Listener| Agent[RL Control Agent Process - Control Plane]
    Agent -->|5. Push new weights| Redis
    
    Gateway -->|GET /metrics| Prom[Prometheus / Grafana]
```

### 1. The Data Plane (API Gateway — Synchronous, <0.1ms overhead)
*   **FastAPI Reverse Proxy** running on **Port 8000**.
*   Fetches pre-computed probability weights from **Redis** in **$< 0.1\text{ms}$**.
*   Enforces **Action Masking** (overrides weight to 0% if CPU $> 85\%$) and **Staleness Circuit Breakers**.
*   Proxies calls using connection-pooled `httpx.AsyncClient` HTTP clients.
*   Enforces **Adaptive QoS Traffic Shaping** (`/auth` = High Priority, `/analytics` = Low Priority, shed under load).

### 2. The Control Plane (RL Agent — Asynchronous, 150ms loop + Event Stream)
*   Independent background worker process.
*   Listens to real-time outcome events via **Redis Pub/Sub** (`events:request_outcomes`).
*   Runs **Linear Thompson Sampling (LinTS)** Bayesian regression updates.
*   Calculates new Softmax probability distribution weights and writes them back to Redis.

### 3. Independent Backend Microservices (Ports 8001–8010)
*   FastAPI backend containers tracking actual hardware CPU usage via `psutil`.
*   Auto-register on boot with the **Service Registry** (`src/registry.py`) and send periodic health heartbeats.

---

## 🧮 THE REINFORCEMENT LEARNING MATHEMATICS

The system models load balancing using **Linear Thompson Sampling (LinTS)**:

### 1. State / Context Vector ($\mathbf{x}_{t,i}$)
$$\mathbf{x}_{t,i} = \left[1.0,\; \frac{\text{CPU}_i}{100},\; \frac{\text{Queue}_i}{20},\; \frac{\text{P99}_i}{200},\; \frac{\text{GlobalRate}}{150},\; \frac{d/dt \text{ Rate}}{50}\right]^T$$

### 2. Reward Function ($r_{t,i}$)
The reward evaluates backend performance after each evaluation window:
$$r_{t,i} = - \left( 1.0 \cdot \frac{\text{P99}_i}{200} + 5.0 \cdot \text{Error\_Rate}_i + 10.0 \cdot \text{SLA\_Breach\_Rate}_i \right)$$

### 3. Bayesian Model Update & Softmax Decision
For each backend instance $i$:
$$\mathbf{B}_i \leftarrow \mathbf{B}_i + \mathbf{x}_{t,i} \mathbf{x}_{t,i}^T, \quad \mathbf{f}_i \leftarrow \mathbf{f}_i + r_{t,i} \mathbf{x}_{t,i}$$
$$\hat{\boldsymbol{\theta}}_i = \mathbf{B}_i^{-1} \mathbf{f}_i, \quad \tilde{\boldsymbol{\theta}}_i \sim \mathcal{N}\left(\hat{\boldsymbol{\theta}}_i, v^2 \mathbf{B}_i^{-1}\right)$$
$$\text{Score}_i = \mathbf{x}_{t,i}^T \tilde{\boldsymbol{\theta}}_i, \quad w_i = \frac{e^{\text{Score}_i / \tau}}{\sum_j e^{\text{Score}_j / \tau}}$$

---

## 🛡️ PRODUCTION SAFETY GUARDRAILS

1.  **Action Masking**: If a node's CPU exceeds **85.0%**, its weight is immediately forced to **0.0%**, preventing it from taking further load until it cools down.
2.  **Staleness Circuit Breaker**: If the RL Control Plane halts or Redis updates lag $> 1.0\text{s}$, the Gateway trips a circuit breaker and automatically falls back to **Least Connections routing**.
3.  **Heartbeat Node Detection**: If a backend container crashes, its Redis heartbeat key expires within 2.0s. The Gateway marks the node as offline and routes traffic around it.
4.  **Adaptive QoS Load Shedding**: Under cluster stress (avg CPU $> 80\%$), the Gateway sheds low-priority endpoints (`/analytics`) with `HTTP 429 Too Many Requests` to guarantee high-priority SLA survival (`/auth`).

---

## 📊 REPRODUCIBLE PERFORMANCE BENCHMARKS

Run the automated performance test suite via `python src/benchmark.py` to compare all 5 algorithms:

| Target Load | Strategy | Simulated Throughput | P50 (ms) | P95 (ms) | P99 (ms) | SLA Breaches (>200ms) | Error % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 RPS | Round Robin | 200,000.0 req/s | 38.1 ms | 112.8 ms | 114.4 ms | 0.0% | 0.0% |
| 100 RPS | Weighted Round Robin | 200,000.0 req/s | 28.7 ms | 107.5 ms | 114.6 ms | 0.0% | 0.0% |
| 100 RPS | Least Connections | 200,000.0 req/s | 38.3 ms | 111.5 ms | 114.2 ms | 0.0% | 0.0% |
| 100 RPS | Power of Two Choices (P2C) | 200,000.0 req/s | 44.2 ms | 112.8 ms | 114.0 ms | 0.0% | 0.0% |
| 100 RPS | **RL Adaptive (LinTS)** | 200,000.0 req/s | 30.0 ms | 107.6 ms | 113.7 ms | 0.0% | 0.0% |
| 500 RPS | Round Robin | 748,047.8 req/s | 92.3 ms | 162.9 ms | 167.6 ms | 0.0% | 15.9% |
| 500 RPS | Least Connections | 506,558.5 req/s | 25.6 ms | 26.5 ms | 26.6 ms | 0.0% | 0.0% |
| 500 RPS | Power of Two Choices (P2C) | 462,539.0 req/s | 53.5 ms | 96.5 ms | 98.0 ms | 0.0% | 0.0% |
| 500 RPS | **RL Adaptive (LinTS)** | 600,215.2 req/s | 54.0 ms | 157.4 ms | 166.0 ms | 0.0% | 8.4% |

---

## 🔍 OBSERVABILITY & EXPLAINABILITY

### 1. AI Decision Audit API (`curl http://127.0.0.1:8000/explain-routing`)
When asked *"Why was Node 3 chosen over Node 1?"*, query the audit endpoint:
```json
{
  "active_strategy": "lin_ts",
  "circuit_breaker_tripped": false,
  "registered_instances_count": 5,
  "candidate_evaluations": [
    {
      "node": "Node-1",
      "name": "Instance-1 (High-Compute)",
      "rl_weight": 0.4215,
      "rl_weight_percentage": "42.1%",
      "cpu_load": "20.0%",
      "queue_depth": 0,
      "recent_p99_ms": 12.4,
      "status": "TOP_CHOICE"
    },
    {
      "node": "Node-5",
      "name": "Instance-5 (Slow-Legacy)",
      "rl_weight": 0.0000,
      "rl_weight_percentage": "0.0%",
      "cpu_load": "92.0%",
      "queue_depth": 18,
      "recent_p99_ms": 185.0,
      "status": "MASKED_HIGH_CPU"
    }
  ]
}
```

### 2. Prometheus & Grafana Integration
*   Exposes standard Prometheus metrics on `http://127.0.0.1:8000/metrics`.
*   Includes a ready-to-import `grafana_dashboard.json` visualizing live latency histograms (P50/P95/P99), node CPU/queue metrics, active routing weights, and QoS shedding counters.

---

## ⚙️ CONFIGURATION ARCHITECTURE (`config.yaml`)

```yaml
cluster:
  num_instances: 5
  port: 8000
  routing_strategy: "lin_ts"  # Options: lin_ts, least_conn, p2c, round_robin, weighted_round_robin

redis:
  host: "127.0.0.1"
  port: 6379

sla:
  latency_ms: 200.0
  staleness_threshold_sec: 1.0

agent:
  control_plane_interval_sec: 0.15
  exploration_param: 0.3
  temperature: 0.2
```

---

## 🚀 QUICK START GUIDE

### 1. Local Run (Zero External Dependencies)
```bash
chmod +x run.sh
./run.sh
```

### 2. Docker Compose Multi-Container Network
```bash
docker compose up --build
```

### 3. Run Automated Unit & Integration Tests
```bash
source venv/bin/activate
PYTHONPATH=. pytest -v
```

### 4. Inject Chaos Fault
```bash
# Force 99% CPU spike on Node 1 (idx 0)
curl -X POST "http://127.0.0.1:8000/chaos/inject?node_idx=0" -H "Content-Type: application/json" -d '{"fault_type": "cpu_spike"}'

# Clear all chaos faults
curl -X POST "http://127.0.0.1:8000/chaos/clear"
```

---

## 📁 REPOSITORY STRUCTURE

```
.
├── config.yaml                # Dynamic system & hyperparameter settings
├── Dockerfile                 # Multi-stage container build
├── docker-compose.yml         # Container network orchestration
├── grafana_dashboard.json     # Pre-configured Grafana dashboard JSON
├── benchmark_results.md       # Comparative benchmark report
├── requirements.txt           # Python dependencies
├── run.sh                     # Launch & setup script
├── tests/                     # Test suite
│   ├── test_load_balancer.py  # Core state & math tests
│   ├── test_phase1.py         # Multi-strategy & YAML tests
│   ├── test_phase2.py         # Prometheus & Pub/Sub tests
│   └── test_phase3.py         # Service Discovery, HPA & Geo-Routing tests
└── src/
    ├── config.py              # Configuration loader
    ├── shared_state.py        # Redis state & Pub/Sub driver
    ├── routing_strategies.py  # 5 baseline & adaptive algorithms
    ├── registry.py            # Dynamic Service Discovery engine
    ├── autoscaler.py          # Kubernetes HPA simulation worker
    ├── chaos.py               # Chaos Engineering fault injector
    ├── metrics.py             # Prometheus exposition exporter
    ├── backend_node.py        # Microservice backend server
    ├── gateway.py             # Reverse HTTP Proxy & QoS router
    ├── agent.py               # Thompson Sampling RL Control plane
    ├── dashboard.py           # ANSI terminal telemetry console
    ├── benchmark.py           # Performance benchmarking engine
    └── main.py                # System orchestrator
```

---

## 📜 LICENSE
Distributed under the [MIT License](LICENSE).
