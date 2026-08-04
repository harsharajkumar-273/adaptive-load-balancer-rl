# Comparative Load Balancer Performance Benchmark Report

| Target RPS | Strategy | Actual Throughput | P50 (ms) | P95 (ms) | P99 (ms) | SLA Breaches (>200ms) | Error % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 RPS | ROUND_ROBIN | 98.8 RPS | 6.7 ms | 11.1 ms | 14.2 ms | 0.0% | 100.0% |
| 100 RPS | WEIGHTED_ROUND_ROBIN | 98.1 RPS | 5.9 ms | 11.4 ms | 13.5 ms | 0.0% | 100.0% |
| 100 RPS | LEAST_CONN | 98.9 RPS | 7.0 ms | 11.4 ms | 21.6 ms | 0.0% | 100.0% |
| 100 RPS | P2C | 98.7 RPS | 6.8 ms | 10.7 ms | 11.4 ms | 0.0% | 100.0% |
| 100 RPS | **RL Adaptive (LinTS)** | 98.8 RPS | 5.5 ms | 17.5 ms | 46.8 ms | 0.0% | 100.0% |
| 500 RPS | ROUND_ROBIN | 494.0 RPS | 13.9 ms | 23.6 ms | 36.7 ms | 0.0% | 100.0% |
| 500 RPS | WEIGHTED_ROUND_ROBIN | 494.3 RPS | 14.4 ms | 31.6 ms | 43.2 ms | 0.0% | 100.0% |
| 500 RPS | LEAST_CONN | 494.3 RPS | 13.2 ms | 24.1 ms | 31.8 ms | 0.0% | 100.0% |
| 500 RPS | P2C | 495.2 RPS | 15.7 ms | 55.7 ms | 88.3 ms | 0.0% | 100.0% |
| 500 RPS | **RL Adaptive (LinTS)** | 494.8 RPS | 14.9 ms | 22.9 ms | 46.2 ms | 0.0% | 100.0% |
| 1000 RPS | ROUND_ROBIN | 989.9 RPS | 22.7 ms | 31.4 ms | 36.3 ms | 0.0% | 100.0% |
| 1000 RPS | WEIGHTED_ROUND_ROBIN | 989.7 RPS | 21.9 ms | 31.6 ms | 37.8 ms | 0.0% | 100.0% |
| 1000 RPS | LEAST_CONN | 989.6 RPS | 22.0 ms | 30.8 ms | 35.6 ms | 0.0% | 100.0% |
| 1000 RPS | P2C | 989.6 RPS | 22.9 ms | 31.0 ms | 35.3 ms | 0.0% | 100.0% |
| 1000 RPS | **RL Adaptive (LinTS)** | 989.5 RPS | 23.4 ms | 31.3 ms | 36.8 ms | 0.0% | 100.0% |
