# AI-Driven Asynchronous Load Balancer using Reinforcement Learning
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fully functional, end-to-end prototype of an **AI-Driven Asynchronous Load Balancer** utilizing **Contextual Multi-Armed Bandits (Linear Thompson Sampling)**. This system is designed to dynamically mitigate sudden, massive thundering-herd traffic spikes (e.g., a Netflix "Stranger Things" release) by shifting routing weights in real-time away from degrading backend clusters to prevent SLA breaches.

---

## 🏗️ System Architecture

To ensure sub-millisecond routing decisions, the AI inference and model updates are completely decoupled from the data plane.

```
                    ┌────────────────────────┐
                    │    Client Requests     │
                    └───────────┬────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │             API GATEWAY (FastAPI)            │
         │  1. Staleness Check (CB)                     │
         │  2. Action Masking (CPU > 85% -> 0 weight)   │
         │  3. Stochastic Routing Decision              │
         └─────────┬──────────────────────────┬─────────┘
                   │                          │
                   ├──────────────────────────┼─ ... (to other backends)
                   ▼                          ▼
      ┌─────────────────────────┐  ┌─────────────────────────┐
      │  Instance 1 (Compute)   │  │  Instance 5 (Legacy)    │
      │  - Base Latency: 10ms   │  │  - Base Latency: 80ms   │
      │  - Capacity: 150 RPS    │  │  - Capacity: 20 RPS     │
      └────────────┬────────────┘  └────────────┬────────────┘
                   │                            │
                   └─────────────┬──────────────┘
                                 │ Writes metrics
                                 ▼
         ┌──────────────────────────────────────────────┐
         │             SHARED MEMORY CACHE              │
         │  - Routing weights & updates                 │
         │  - Instance CPU, Queue, Latency, Errors      │
         └──────────────────────▲───────────────────────┘
                                │
                      Pulls     │ Pushes
                      Metrics   │ Weights
                                │
         ┌──────────────────────┴───────────────────────┐
         │              RL CONTROL AGENT                │
         │  - Runs Contextual Bandit loop (150ms)       │
         │  - Updates Linear Thompson Sampling weights   │
         └──────────────────────────────────────────────┘
```

1.  **THE DATA PLANE (Synchronous)**: A fast FastAPI gateway that receives client requests, fetches the latest routing weights from local cache memory under **0.1ms**, checks safety guardrails, stochastically routes the request, and returns the response.
2.  **THE CONTROL PLANE (Asynchronous)**: A background worker that runs every **150ms**. It flushes accumulated data plane feedback metrics, computes rewards, updates the Linear Thompson Sampling (LinTS) regression parameters, samples new routing weights, and updates the local cache.

---

## 🧮 Reinforcement Learning Formulation: Linear Thompson Sampling (LinTS)

The routing decision is formulated as a Contextual Multi-Armed Bandit where each backend instance represents an action.

### 1. State Space (Context $x_{t,i}$)
For each backend instance $i$, the agent constructs a 6-dimensional context vector:
$$x_{t,i} = \begin{bmatrix} 
1.0 & \text{(Bias/Baseline)} \\
\text{CPU}_i / 100 & \text{(Normalized CPU Load)} \\
\text{QueueDepth}_i / 20 & \text{(Normalized Queue Depth)} \\
\text{P99\_Latency}_i / 200 & \text{(Normalized Recent SLA Latency)} \\
\text{GlobalRate} / 150 & \text{(Normalized Traffic Volatility)} \\
\frac{d}{dt}\text{Rate} / 50 & \text{(Traffic Volatility Trend)}
\end{bmatrix}^T$$

### 2. Bayesian Regression Parameters
The agent models the expected reward parameters for each instance $i$ using a Gaussian posterior:
*   $\mathbf{B}_i$: A $6 \times 6$ precision matrix (initially $\mathbf{I}$).
*   $f_i$: A $6$-dimensional accumulated reward vector (initially $\mathbf{0}$).
*   $\hat{\theta}_i$: Mean coefficient vector $\mathbf{B}_i^{-1} f_i$.

At each interval, the agent samples a coefficient vector $\tilde{\theta}_i$ from the posterior distribution:
$$\tilde{\theta}_i \sim \mathcal{N}(\hat{\theta}_i, v^2 \mathbf{B}_i^{-1})$$
where $v^2$ is the exploration factor (`RL_EXPLORATION_PARAM = 0.3`).

### 3. Decision Rule & Probability Distribution
The expected score (reward) is computed for each instance:
$$\text{Score}_i = x_{t,i}^T \tilde{\theta}_i$$
To distribute incoming traffic stochastically across the cluster, routing weights are mapped using a Softmax function over the expected scores:
$$w_i = \frac{e^{\text{Score}_i / \tau}}{\sum_j e^{\text{Score}_j / \tau}}$$
where $\tau$ is the routing temperature (`RL_TEMPERATURE = 0.2`).

### 4. Reward Function
Every 150ms, feedback is evaluated based on the performance of instances that handled requests:
$$r_{t,i} = - \left( 1.0 \cdot \left(\frac{\text{P99}_i}{200}\right) + 5.0 \cdot \text{Error\_Rate}_i + 10.0 \cdot \text{SLA\_Breach\_Rate}_i \right)$$
*   *SLA Breach* is defined as request latency exceeding **200ms**.
*   The reward is a negative penalty (higher is better, i.e., closer to 0).

---

## 🛡️ Safety Guardrails

To protect the downstream service cluster from cascading failures, the Data Plane gateway implements two hardcoded safety guardrails:

*   **Action Masking**: If any instance CPU utilization exceeds **85.0%**, the gateway overrides its weight to **0.0** and renormalizes the remaining active instances. If all instances are overloaded, it defaults to Least Connections.
*   **Circuit Breaker (Cache Staleness)**: If the asynchronous Control Plane loop fails to update the weights cache in over **1.0 second** (indicating a thread hang, Python garbage collection pause, or crash), the gateway trips the circuit breaker and falls back to standard **Least Connections** routing.

---

## 💾 Checkpointing (Persistent Learning)

To avoid having the load balancer start from scratch on every startup:
*   **On Exit**: When the system is gracefully shut down using `Ctrl + C`, the orchestrator invokes `save_checkpoint()` to serialize precision matrices $B_i$ and vectors $f_i$ into `model_checkpoint.npz` using NumPy's binary compressed formatting.
*   **On Startup**: The orchestrator checks if `model_checkpoint.npz` exists, restores the matrices, and computes $\hat{\theta}_i = B_i^{-1} f_i$, immediately resuming routing decisions based on prior learning.

---

## 📁 File Structure

```
.
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── run.sh                     # Launch and auto-installation script
├── model_checkpoint.npz       # Saved agent checkpoint (auto-generated)
├── stdout.log                 # Console output redirect target
└── src/
    ├── config.py              # Configuration and backend specifications
    ├── shared_state.py        # Thread-safe in-memory cache
    ├── agent.py               # Linear Thompson Sampling RL Control Plane
    ├── gateway.py             # FastAPI API Gateway Data Plane
    ├── dashboard.py           # ANSI terminal dashboard display
    └── main.py                # Orchestrator and entry point
```

---

## 🚀 Installation & Running

### Prerequisites
*   Python 3.10+
*   Ports: Port `8000` must be free.

### Step 1: Clone and Navigate
Clone the code to your workspace and navigate to the project directory:
```bash
cd "/Users/harsharajkumar/Downloads/projects/netflix rl"
```

### Step 2: Run Auto-Startup Script
Run the pre-configured startup script. It will automatically check for Python, set up a virtual environment, install the requirements, and boot up the system:
```bash
chmod +x run.sh
./run.sh
```

*(Alternatively, you can run manually by setting up a `venv`, installing packages from `requirements.txt`, and running `python3 src/main.py`)*

---

## 📊 Live Telemetry Dashboard & Simulation Timeline

The simulation executes a repeating **60-second traffic pattern** to demonstrate both peak performance handling and failovers:

1.  **Base Load (0s - 10s)**: Generates 15 RPS. All backends are healthy.
2.  **Surge Spike (10s - 20s)**: Traffic climbs exponentially up to **160 RPS** (representing a movie release). The RL agent shifts weights to Instance 1 and 2, while masking Instance 5 to 0% to prevent SLA breaches.
3.  **Peak Surge (20s - 28s)**: Holds at 160 RPS.
4.  **Mock Control Plane Crash (30s - 36s)**: The RL Agent loop is programmatically stopped. Within 1.0 second, the dashboard will display `Circuit Breaker State: TRIPPED` in red, and the gateway will fallback to **Least Connections routing**.
5.  **Recovery (36s - 45s)**: The agent loop restarts, the circuit breaker resets to `CLOSED` (green), and RL routing resumes as traffic subsides.
6.  **Cooldown (45s - 60s)**: Returns to base load.

---

## 🧪 Diagnostics & Overfitting Verification

To verify that the model has compiled parameters accurately and is not experiencing **overfitting/overconfidence lock-in** (where parameter variances drop to absolute 0, stopping all exploration), we inspect the saved checkpoint:

```bash
# Run the diagnostic script to analyze checkpoint values
python3 -c "
import numpy as np
data = np.load('model_checkpoint.npz')
for i in range(5):
    cov = np.linalg.inv(data['B'][i])
    uncertainties = np.sqrt(np.diag(cov))
    print(f'Instance {i+1} CPU Uncertainty: {uncertainties[1]:.4f}')
"
```

If the parameter uncertainty (Std Dev) is $>0.05$ (like in our model), the Thompson Sampling algorithm will continue to explore other routing branches dynamically, proving that the model is in a healthy, adaptive state.
