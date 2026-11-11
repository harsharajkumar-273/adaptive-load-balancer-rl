# Trust gate: offline replay of the two grade lists (ungated routers, 3 seeds per cell)

Gate rule: trust iff over the last `w` graded steps the model is within tolerance more often than the baseline (persistence: the server's smoothed own latency). `caught` = share of the worst 1% of latencies that occurred while the gate had withdrawn trust; `taken` = share of all steps the gate would have taken away. A useless gate has caught = taken.


## linear model

| cell | w | model ok | baseline ok | taken | caught | lift |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| steady 50 ms | 10 | 0.48 | 0.48 | 0.65 | 0.76 | 1.18x |
| steady 50 ms | 25 | 0.48 | 0.48 | 0.56 | 0.66 | 1.17x |
| steady 50 ms | 50 | 0.48 | 0.48 | 0.52 | 0.62 | 1.19x |
| steady 50 ms | 100 | 0.48 | 0.48 | 0.50 | 0.66 | 1.32x |
| steady 50 ms | 200 | 0.48 | 0.48 | 0.48 | 0.73 | 1.51x |
| steady 200 ms | 10 | 0.43 | 0.43 | 0.66 | 0.65 | 0.98x |
| steady 200 ms | 25 | 0.43 | 0.43 | 0.57 | 0.47 | 0.83x |
| steady 200 ms | 50 | 0.43 | 0.43 | 0.52 | 0.37 | 0.71x |
| steady 200 ms | 100 | 0.43 | 0.43 | 0.50 | 0.33 | 0.65x |
| steady 200 ms | 200 | 0.43 | 0.43 | 0.49 | 0.28 | 0.58x |
| gray 50 ms | 10 | 0.47 | 0.46 | 0.64 | 0.72 | 1.14x |
| gray 50 ms | 25 | 0.47 | 0.46 | 0.55 | 0.64 | 1.15x |
| gray 50 ms | 50 | 0.47 | 0.46 | 0.52 | 0.64 | 1.25x |
| gray 50 ms | 100 | 0.47 | 0.46 | 0.50 | 0.63 | 1.25x |
| gray 50 ms | 200 | 0.47 | 0.46 | 0.49 | 0.58 | 1.19x |
| gray 200 ms | 10 | 0.39 | 0.39 | 0.67 | 0.70 | 1.05x |
| gray 200 ms | 25 | 0.39 | 0.39 | 0.58 | 0.63 | 1.08x |
| gray 200 ms | 50 | 0.39 | 0.39 | 0.56 | 0.63 | 1.13x |
| gray 200 ms | 100 | 0.39 | 0.39 | 0.56 | 0.63 | 1.13x |
| gray 200 ms | 200 | 0.39 | 0.39 | 0.58 | 0.61 | 1.04x |

## quadratic model

| cell | w | model ok | baseline ok | taken | caught | lift |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| steady 50 ms | 10 | 0.37 | 0.36 | 0.65 | 0.72 | 1.10x |
| steady 50 ms | 25 | 0.37 | 0.36 | 0.56 | 0.65 | 1.16x |
| steady 50 ms | 50 | 0.37 | 0.36 | 0.52 | 0.61 | 1.18x |
| steady 50 ms | 100 | 0.37 | 0.36 | 0.47 | 0.60 | 1.28x |
| steady 50 ms | 200 | 0.37 | 0.36 | 0.44 | 0.59 | 1.34x |
| steady 200 ms | 10 | 0.32 | 0.33 | 0.67 | 0.65 | 0.97x |
| steady 200 ms | 25 | 0.32 | 0.33 | 0.59 | 0.52 | 0.89x |
| steady 200 ms | 50 | 0.32 | 0.33 | 0.57 | 0.51 | 0.90x |
| steady 200 ms | 100 | 0.32 | 0.33 | 0.57 | 0.51 | 0.90x |
| steady 200 ms | 200 | 0.32 | 0.33 | 0.58 | 0.50 | 0.86x |
| gray 50 ms | 10 | 0.38 | 0.36 | 0.64 | 0.56 | 0.87x |
| gray 50 ms | 25 | 0.38 | 0.36 | 0.56 | 0.43 | 0.76x |
| gray 50 ms | 50 | 0.38 | 0.36 | 0.53 | 0.33 | 0.63x |
| gray 50 ms | 100 | 0.38 | 0.36 | 0.50 | 0.28 | 0.56x |
| gray 50 ms | 200 | 0.38 | 0.36 | 0.50 | 0.08 | 0.17x |
| gray 200 ms | 10 | 0.30 | 0.31 | 0.65 | 0.71 | 1.09x |
| gray 200 ms | 25 | 0.30 | 0.31 | 0.58 | 0.63 | 1.08x |
| gray 200 ms | 50 | 0.30 | 0.31 | 0.56 | 0.54 | 0.96x |
| gray 200 ms | 100 | 0.30 | 0.31 | 0.56 | 0.50 | 0.89x |
| gray 200 ms | 200 | 0.30 | 0.31 | 0.60 | 0.42 | 0.70x |

## ewma model

| cell | w | model ok | baseline ok | taken | caught | lift |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| steady 50 ms | 10 | 0.54 | 0.51 | 0.60 | 0.80 | 1.34x |
| steady 50 ms | 25 | 0.54 | 0.51 | 0.50 | 0.73 | 1.46x |
| steady 50 ms | 50 | 0.54 | 0.51 | 0.44 | 0.66 | 1.50x |
| steady 50 ms | 100 | 0.54 | 0.51 | 0.38 | 0.59 | 1.54x |
| steady 50 ms | 200 | 0.54 | 0.51 | 0.32 | 0.56 | 1.77x |
| steady 200 ms | 10 | 0.44 | 0.44 | 0.67 | 0.72 | 1.07x |
| steady 200 ms | 25 | 0.44 | 0.44 | 0.57 | 0.58 | 1.01x |
| steady 200 ms | 50 | 0.44 | 0.44 | 0.53 | 0.51 | 0.96x |
| steady 200 ms | 100 | 0.44 | 0.44 | 0.50 | 0.43 | 0.87x |
| steady 200 ms | 200 | 0.44 | 0.44 | 0.46 | 0.32 | 0.70x |
| gray 50 ms | 10 | 0.51 | 0.48 | 0.60 | 0.77 | 1.28x |
| gray 50 ms | 25 | 0.51 | 0.48 | 0.49 | 0.66 | 1.36x |
| gray 50 ms | 50 | 0.51 | 0.48 | 0.43 | 0.69 | 1.58x |
| gray 50 ms | 100 | 0.51 | 0.48 | 0.37 | 0.78 | 2.08x |
| gray 50 ms | 200 | 0.51 | 0.48 | 0.31 | 0.82 | 2.62x |
| gray 200 ms | 10 | 0.43 | 0.43 | 0.65 | 0.74 | 1.13x |
| gray 200 ms | 25 | 0.43 | 0.43 | 0.56 | 0.59 | 1.05x |
| gray 200 ms | 50 | 0.43 | 0.43 | 0.52 | 0.50 | 0.96x |
| gray 200 ms | 100 | 0.43 | 0.43 | 0.50 | 0.43 | 0.85x |
| gray 200 ms | 200 | 0.43 | 0.43 | 0.49 | 0.43 | 0.88x |