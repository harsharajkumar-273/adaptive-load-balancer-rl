# Distributed AI-Driven Load Balancer (Redis + FastAPI Microservices)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-7.0+-red.svg)](https://redis.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Exporter-orange.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-orange.svg)](https://grafana.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

A portfolio-defining, enterprise-grade prototype of an **AI-Driven Asynchronous Load Balancer** implemented as a **distributed microservice cluster**. Features **multi-strategy routing baselines**, **Prometheus observability & Grafana dashboards**, **Redis Pub/Sub reactive streaming**, **Adaptive QoS Traffic Shaping**, **AI routing explainability**, and **chaos engineering fault injection**.

---

## 🏗️ System Architecture

```mermaid
graph TD
    Client[Traffic Generator / Load Tester] -->|HTTP:8000/auth, /play-video, /analytics| Gateway[API Gateway Proxy]
    Gateway -->|1. Fetch Weights & Telemetry| Redis[(Redis State Store)]
    Gateway -->|2. Route via lin_ts / least_conn / p2c / rr| Node1[Backend Node 1 - Port 8001]
    Gateway -->|2. Route via lin_ts / least_conn / p2c / rr| Node2[Backend Node 2 - Port 8002]
    Gateway -->|2. Route via lin_ts / least_conn / p2c / rr| Node5[Backend Node 5 - Port 8005]
    
    Node1 -->|Heartbeat & CPU Metrics| Redis
    Node2 -->|Heartbeat & CPU Metrics| Redis
    Node5 -->|Heartbeat & CPU Metrics| Redis
    
    Gateway -->|3. Publish Outcome Stream (events:outcomes)| PubSub[Redis Pub/Sub Channel]
    PubSub -->|4. Reactive Event Listener| Agent[RL Control Agent Process]
    Agent -->|5. Push new weights| Redis
    
    Gateway -->|GET /metrics| Prom[Prometheus / Grafana]
```

1.  **API Gateway Proxy (Port 8000)**: Evaluates routing strategies in **< 0.1ms**, checks safety guardrails, enforces **Adaptive QoS Traffic Shaping**, and publishes outcome events to Redis Pub/Sub.
2.  **Adaptive QoS Traffic Shaping**:
    *   `GET /auth` & `GET /checkout`: **High Priority** (SLA protected).
    *   `GET /play-video`: **Medium Priority**.
    *   `GET /analytics` & `GET /logs`: **Low Priority** (Shed via `HTTP 429` under average cluster CPU > 80%).
3.  **Event-Driven Reactive Streaming (Redis Pub/Sub)**: Eliminates polling delays! Requests publish outcome events to `events:request_outcomes`. The RL Agent listens reactively to update parameters on-the-fly.
4.  **Prometheus & Grafana Observability**: Exposes standard `/metrics` exposition format. Includes pre-configured `grafana_dashboard.json`.
5.  **Multi-Strategy Routing Engine**: 5 swappable load balancing algorithms (`lin_ts`, `least_conn`, `p2c`, `round_robin`, `weighted_round_robin`).
6.  **AI Routing Explainability (`GET /explain-routing`)**: Audit API detailing candidate evaluations, CPU/queue features, expected rewards, and action mask states.
7.  **Chaos Engineering Engine (`POST /chaos/inject`)**: Injects CPU spikes (99%), extra latency delays, or HTTP 500 error spikes.

---

## 📊 Comparative Performance Benchmark Report

| Target Load | Strategy | Simulated Throughput | P50 (ms) | P95 (ms) | P99 (ms) | SLA Breaches (>200ms) | Error % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 RPS | Round Robin | 200,000.0 req/s | 38.1 ms | 112.8 ms | 114.4 ms | 0.0% | 0.0% |
| 100 RPS | Weighted Round Robin | 200,000.0 req/s | 28.7 ms | 107.5 ms | 114.6 ms | 0.0% | 0.0% |
| 100 RPS | Least Connections | 200,000.0 req/s | 38.3 ms | 111.5 ms | 114.2 ms | 0.0% | 0.0% |
| 100 RPS | Power of Two Choices (P2C) | 200,000.0 req/s | 44.2 ms | 112.8 ms | 114.0 ms | 0.0% | 0.0% |
| 100 RPS | **RL Adaptive (LinTS)** | 200,000.0 req/s | 30.0 ms | 107.6 ms | 113.7 ms | 0.0% | 0.0% |
| 500 RPS | Least Connections | 506,558.5 req/s | 25.6 ms | 26.5 ms | 26.6 ms | 0.0% | 0.0% |
| 500 RPS | Power of Two Choices (P2C) | 462,539.0 req/s | 53.5 ms | 96.5 ms | 98.0 ms | 0.0% | 0.0% |
| 500 RPS | **RL Adaptive (LinTS)** | 600,215.2 req/s | 54.0 ms | 157.4 ms | 166.0 ms | 0.0% | 8.4% |

---

## ⚙️ Prometheus & Grafana Configuration

### Prometheus Scraping Target
Exposed on `http://127.0.0.1:8000/metrics`
```yaml
scrape_configs:
  - job_name: 'netflix_rl_load_balancer'
    scrape_interval: 1s
    static_configs:
      - targets: ['localhost:8000']
```

### Import Grafana Dashboard
Import `grafana_dashboard.json` directly into Grafana to visualize live P50/P95/P99 latency histograms, active routing weights, node CPU/queue depths, circuit breaker states, and QoS shedding counters!

---

## 🚀 Execution & Testing

### 1. Run Cluster Locally
```bash
./run.sh
```

### 2. Run Docker Compose Network
```bash
docker compose up --build
```

### 3. Run Automated Testing Suite
```bash
source venv/bin/activate
PYTHONPATH=. pytest -v
```

### 4. Test QoS Traffic Shedding
```bash
# High Priority - Guaranteed SLA
curl http://127.0.0.1:8000/auth

# Low Priority - Sheds via HTTP 429 under CPU > 80%
curl http://127.0.0.1:8000/analytics
```
