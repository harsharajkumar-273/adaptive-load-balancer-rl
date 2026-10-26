# Agent-overload study: results

Median over 10 seeds unless stated; goodput = completed steps / no-failure step capacity.

## Overload sweep: goodput (completed steps / capacity) (median over seeds)

| policy | 0.6 | 0.9 | 1.2 | 1.6 | 2.4 | 4.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| none | 0.58 | 0.87 | 0.37 | 0.31 | 0.26 | 0.22 |
| prog | 0.58 | 0.87 | 1.01 | 0.64 | 0.16 | 0.08 |
| shed|q=16 | 0.58 | 0.87 | 0.87 | 0.67 | 0.53 | 0.46 |
| cap|n=64 | 0.58 | 0.87 | 1.03 | 1.01 | 1.02 | 1.02 |
| task|length=mean|h=20 | 0.58 | 0.87 | 1.01 | 1.01 | 1.03 | 1.03 |
| little|d=5 | 0.58 | 0.87 | 1.04 | 1.02 | 1.03 | 1.02 |
| little|d=5|ident=infer|p_id=0.7 | 0.58 | 0.87 | 1.03 | 1.00 | 1.00 | 1.00 |
| cap|n=128 | 0.58 | 0.87 | 1.02 | 0.92 | 0.90 | 0.91 |
| cap|n=16 | 0.43 | 0.45 | 0.45 | 0.46 | 0.45 | 0.45 |
| cap|n=32 | 0.58 | 0.82 | 0.87 | 0.88 | 0.89 | 0.89 |
| little|d=1 | 0.58 | 0.87 | 1.02 | 1.00 | 1.02 | 1.02 |
| little|d=10 | 0.58 | 0.87 | 1.04 | 1.02 | 1.03 | 1.02 |
| little|d=2.5 | 0.58 | 0.87 | 1.03 | 1.02 | 1.02 | 1.02 |
| little|d=5|fifo=1 | 0.58 | 0.87 | 1.04 | 1.02 | 1.03 | 1.02 |
| little|d=5|hint=1 | 0.58 | 0.87 | 1.04 | 1.02 | 1.03 | 1.02 |
| little|d=5|ident=infer|p_id=0.5 | 0.58 | 0.87 | 1.01 | 0.96 | 0.92 | 0.90 |
| little|d=5|ident=infer|p_id=0.9 | 0.58 | 0.87 | 1.04 | 1.01 | 1.02 | 1.00 |
| shed|q=4 | 0.58 | 0.86 | 0.91 | 0.74 | 0.57 | 0.45 |
| shed|q=64 | 0.58 | 0.87 | 0.63 | 0.50 | 0.49 | 0.45 |
| task|length=mean|h=10 | 0.58 | 0.76 | 0.79 | 0.81 | 0.82 | 0.82 |
| task|length=mean|h=40 | 0.58 | 0.87 | 1.05 | 1.01 | 1.01 | 1.01 |
| task|length=mean|h=5 | 0.38 | 0.38 | 0.39 | 0.39 | 0.39 | 0.39 |
| task|length=mean|h=80 | 0.58 | 0.87 | 1.03 | 0.93 | 0.90 | 0.92 |
| task|length=oracle|h=10 | 0.44 | 0.56 | 0.64 | 0.73 | 0.87 | 0.99 |
| task|length=oracle|h=20 | 0.57 | 0.77 | 0.90 | 1.00 | 1.02 | 1.03 |
| task|length=oracle|h=40 | 0.58 | 0.87 | 1.01 | 1.00 | 0.99 | 0.91 |
| task|length=pred|sigma=0.5|h=10 | 0.44 | 0.56 | 0.66 | 0.78 | 0.93 | 1.02 |
| task|length=pred|sigma=0.5|h=20 | 0.53 | 0.74 | 0.88 | 1.00 | 1.01 | 1.02 |
| task|length=pred|sigma=0.5|h=40 | 0.58 | 0.86 | 1.01 | 1.01 | 0.95 | 0.84 |

## Overload sweep: tasks lost mid-way (median over seeds)

| policy | 0.6 | 0.9 | 1.2 | 1.6 | 2.4 | 4.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| none | 0.00 | 0.00 | 0.56 | 0.76 | 0.87 | 0.94 |
| prog | 0.00 | 0.00 | 0.17 | 0.61 | 0.93 | 0.98 |
| shed|q=16 | 0.00 | 0.00 | 0.12 | 0.30 | 0.38 | 0.38 |
| cap|n=64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=mean|h=20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=5|ident=infer|p_id=0.7 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 |
| cap|n=128 | 0.00 | 0.00 | 0.12 | 0.22 | 0.19 | 0.12 |
| cap|n=16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cap|n=32 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=2.5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=5|fifo=1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=5|hint=1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| little|d=5|ident=infer|p_id=0.5 | 0.00 | 0.00 | 0.02 | 0.05 | 0.06 | 0.05 |
| little|d=5|ident=infer|p_id=0.9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shed|q=4 | 0.00 | 0.01 | 0.12 | 0.27 | 0.38 | 0.39 |
| shed|q=64 | 0.00 | 0.00 | 0.23 | 0.35 | 0.37 | 0.37 |
| task|length=mean|h=10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=mean|h=40 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 |
| task|length=mean|h=5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=mean|h=80 | 0.00 | 0.00 | 0.12 | 0.21 | 0.17 | 0.11 |
| task|length=oracle|h=10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=oracle|h=20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=oracle|h=40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| task|length=pred|sigma=0.5|h=10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=pred|sigma=0.5|h=20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=pred|sigma=0.5|h=40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.06 |

## Overload sweep: server time wasted (median over seeds)

| policy | 0.6 | 0.9 | 1.2 | 1.6 | 2.4 | 4.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| none | -0.00 | 0.00 | 0.58 | 0.67 | 0.75 | 0.81 |
| prog | -0.00 | 0.00 | 0.09 | 0.37 | 0.72 | 0.83 |
| shed|q=16 | -0.00 | 0.00 | 0.10 | 0.28 | 0.45 | 0.55 |
| cap|n=64 | -0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| task|length=mean|h=20 | -0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 |
| little|d=5 | -0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 |
| little|d=5|ident=infer|p_id=0.7 | -0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.02 |
| cap|n=128 | -0.00 | 0.00 | 0.08 | 0.12 | 0.13 | 0.13 |
| cap|n=16 | 0.00 | -0.00 | 0.00 | 0.00 | -0.00 | -0.00 |
| cap|n=32 | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 | -0.00 |
| little|d=1 | -0.00 | -0.00 | -0.00 | 0.00 | -0.00 | 0.00 |
| little|d=10 | -0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 |
| little|d=2.5 | -0.00 | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 |
| little|d=5|fifo=1 | -0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 |
| little|d=5|hint=1 | -0.00 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 |
| little|d=5|ident=infer|p_id=0.5 | -0.00 | 0.00 | 0.02 | 0.07 | 0.12 | 0.15 |
| little|d=5|ident=infer|p_id=0.9 | -0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shed|q=4 | 0.00 | 0.01 | 0.10 | 0.27 | 0.44 | 0.56 |
| shed|q=64 | -0.00 | 0.00 | 0.18 | 0.34 | 0.44 | 0.53 |
| task|length=mean|h=10 | -0.00 | -0.00 | 0.00 | 0.00 | -0.00 | -0.00 |
| task|length=mean|h=40 | -0.00 | 0.00 | 0.02 | 0.02 | 0.02 | 0.02 |
| task|length=mean|h=5 | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 | 0.00 |
| task|length=mean|h=80 | -0.00 | 0.00 | 0.07 | 0.12 | 0.14 | 0.13 |
| task|length=oracle|h=10 | 0.00 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 |
| task|length=oracle|h=20 | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 | -0.00 |
| task|length=oracle|h=40 | -0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.12 |
| task|length=pred|sigma=0.5|h=10 | 0.00 | 0.00 | -0.00 | 0.00 | -0.00 | -0.00 |
| task|length=pred|sigma=0.5|h=20 | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | -0.00 |
| task|length=pred|sigma=0.5|h=40 | -0.00 | 0.00 | -0.00 | 0.01 | 0.08 | 0.18 |

## Overload sweep: P99 task latency (s) of completed tasks (median over seeds)

| policy | 0.6 | 0.9 | 1.2 | 1.6 | 2.4 | 4.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| none | 132.41 | 136.59 | 323.04 | 373.51 | 334.71 | 311.50 |
| prog | 134.98 | 132.79 | 189.68 | 231.40 | 241.20 | 250.20 |
| shed|q=16 | 132.41 | 134.95 | 196.44 | 185.76 | 138.88 | 128.17 |
| cap|n=64 | 134.98 | 132.79 | 161.19 | 169.10 | 167.48 | 159.58 |
| task|length=mean|h=20 | 134.98 | 133.93 | 152.18 | 157.51 | 151.57 | 143.25 |
| little|d=5 | 132.41 | 136.59 | 211.44 | 217.74 | 214.16 | 214.43 |
| little|d=5|ident=infer|p_id=0.7 | 125.83 | 140.73 | 204.93 | 205.13 | 223.20 | 206.20 |
| cap|n=128 | 134.98 | 132.79 | 191.98 | 199.05 | 194.98 | 188.28 |
| cap|n=16 | 125.59 | 144.48 | 129.17 | 150.86 | 133.33 | 147.55 |
| cap|n=32 | 134.82 | 128.64 | 140.42 | 141.69 | 137.96 | 137.11 |
| little|d=1 | 132.41 | 133.73 | 177.83 | 185.06 | 189.61 | 179.69 |
| little|d=10 | 132.41 | 136.59 | 211.44 | 217.74 | 214.16 | 214.43 |
| little|d=2.5 | 132.41 | 135.54 | 195.92 | 212.33 | 199.63 | 215.00 |
| little|d=5|fifo=1 | 132.41 | 136.59 | 211.44 | 217.74 | 214.16 | 214.43 |
| little|d=5|hint=1 | 132.41 | 136.59 | 211.44 | 217.74 | 214.16 | 214.43 |
| little|d=5|ident=infer|p_id=0.5 | 125.83 | 140.73 | 209.75 | 208.57 | 186.14 | 177.44 |
| little|d=5|ident=infer|p_id=0.9 | 125.83 | 140.73 | 205.44 | 225.79 | 221.22 | 209.31 |
| shed|q=4 | 136.72 | 139.31 | 157.89 | 151.51 | 128.56 | 117.63 |
| shed|q=64 | 132.41 | 136.59 | 336.72 | 295.21 | 229.59 | 196.70 |
| task|length=mean|h=10 | 128.02 | 130.31 | 133.96 | 138.41 | 132.40 | 133.57 |
| task|length=mean|h=40 | 134.98 | 132.79 | 172.72 | 177.40 | 176.83 | 170.44 |
| task|length=mean|h=5 | 123.23 | 143.03 | 139.19 | 147.58 | 137.83 | 131.27 |
| task|length=mean|h=80 | 134.98 | 132.79 | 189.01 | 200.87 | 194.82 | 199.46 |
| task|length=oracle|h=10 | 96.46 | 82.40 | 73.86 | 67.02 | 57.91 | 50.75 |
| task|length=oracle|h=20 | 120.38 | 117.97 | 101.92 | 85.47 | 70.50 | 59.85 |
| task|length=oracle|h=40 | 134.98 | 131.33 | 128.56 | 107.62 | 103.10 | 112.32 |
| task|length=pred|sigma=0.5|h=10 | 100.36 | 94.92 | 86.48 | 83.30 | 74.13 | 66.47 |
| task|length=pred|sigma=0.5|h=20 | 121.89 | 112.77 | 106.78 | 100.54 | 87.59 | 76.18 |
| task|length=pred|sigma=0.5|h=40 | 134.41 | 128.97 | 128.64 | 125.09 | 130.73 | 133.32 |

## Overload sweep: fraction of arriving tasks completed (median over seeds)

| policy | 0.6 | 0.9 | 1.2 | 1.6 | 2.4 | 4.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| none | 1.00 | 1.00 | 0.44 | 0.24 | 0.12 | 0.06 |
| prog | 1.00 | 1.00 | 0.83 | 0.39 | 0.07 | 0.02 |
| shed|q=16 | 1.00 | 1.00 | 0.84 | 0.59 | 0.39 | 0.24 |
| cap|n=64 | 1.00 | 1.00 | 0.88 | 0.63 | 0.42 | 0.25 |
| task|length=mean|h=20 | 1.00 | 1.00 | 0.86 | 0.63 | 0.42 | 0.25 |
| little|d=5 | 1.00 | 1.00 | 0.87 | 0.63 | 0.42 | 0.25 |
| little|d=5|ident=infer|p_id=0.7 | 1.00 | 1.00 | 0.87 | 0.63 | 0.41 | 0.26 |
| cap|n=128 | 1.00 | 1.00 | 0.85 | 0.57 | 0.36 | 0.23 |
| cap|n=16 | 0.77 | 0.51 | 0.38 | 0.27 | 0.18 | 0.11 |
| cap|n=32 | 1.00 | 0.93 | 0.76 | 0.55 | 0.36 | 0.22 |
| little|d=1 | 1.00 | 1.00 | 0.85 | 0.62 | 0.42 | 0.25 |
| little|d=10 | 1.00 | 1.00 | 0.87 | 0.63 | 0.42 | 0.25 |
| little|d=2.5 | 1.00 | 1.00 | 0.86 | 0.63 | 0.42 | 0.25 |
| little|d=5|fifo=1 | 1.00 | 1.00 | 0.87 | 0.63 | 0.42 | 0.25 |
| little|d=5|hint=1 | 1.00 | 1.00 | 0.87 | 0.63 | 0.42 | 0.25 |
| little|d=5|ident=infer|p_id=0.5 | 1.00 | 1.00 | 0.88 | 0.63 | 0.43 | 0.25 |
| little|d=5|ident=infer|p_id=0.9 | 1.00 | 1.00 | 0.88 | 0.62 | 0.42 | 0.25 |
| shed|q=4 | 1.00 | 0.99 | 0.84 | 0.63 | 0.41 | 0.25 |
| shed|q=64 | 1.00 | 1.00 | 0.66 | 0.46 | 0.35 | 0.23 |
| task|length=mean|h=10 | 1.00 | 0.87 | 0.67 | 0.50 | 0.33 | 0.20 |
| task|length=mean|h=40 | 1.00 | 1.00 | 0.87 | 0.62 | 0.41 | 0.25 |
| task|length=mean|h=5 | 0.64 | 0.43 | 0.33 | 0.24 | 0.16 | 0.10 |
| task|length=mean|h=80 | 1.00 | 1.00 | 0.85 | 0.57 | 0.37 | 0.23 |
| task|length=oracle|h=10 | 0.92 | 0.85 | 0.81 | 0.75 | 0.67 | 0.55 |
| task|length=oracle|h=20 | 0.99 | 0.96 | 0.91 | 0.85 | 0.72 | 0.57 |
| task|length=oracle|h=40 | 1.00 | 1.00 | 0.95 | 0.85 | 0.70 | 0.52 |
| task|length=pred|sigma=0.5|h=10 | 0.89 | 0.83 | 0.79 | 0.74 | 0.65 | 0.52 |
| task|length=pred|sigma=0.5|h=20 | 0.97 | 0.94 | 0.90 | 0.83 | 0.68 | 0.52 |
| task|length=pred|sigma=0.5|h=40 | 1.00 | 0.99 | 0.94 | 0.83 | 0.65 | 0.45 |

## Parameter sensitivity at load 2.4 (median goodput)

| rule | value | goodput |
|:---|:---:|:---:|
| shed per-request queue limit Q | 4 | 0.57 |
| shed per-request queue limit Q | 16 | 0.53 |
| shed per-request queue limit Q | 64 | 0.49 |
| cap task cap N | 16 | 0.45 |
| cap task cap N | 32 | 0.89 |
| cap task cap N | 64 | 1.02 |
| cap task cap N | 128 | 0.90 |
| task work budget H (s) | 5 | 0.39 |
| task work budget H (s) | 10 | 0.82 |
| task work budget H (s) | 20 | 1.03 |
| task work budget H (s) | 40 | 1.01 |
| task work budget H (s) | 80 | 0.90 |
| little wait target D (s) | 1 | 1.02 |
| little wait target D (s) | 2.5 | 1.02 |
| little wait target D (s) | 5 | 1.03 |
| little wait target D (s) | 10 | 1.03 |

## Value and cost of length information (median)

| policy | load | work goodput | task goodput | long-task completion | tasks completed |
|:---|:---:|:---:|:---:|:---:|:---:|
| little|d=5 | 1.6 | 1.02 | 1.00 | 0.62 | 0.63 |
| little|d=5 | 4 | 1.02 | 1.03 | 0.24 | 0.25 |
| task|length=mean|h=20 | 1.6 | 1.01 | 1.02 | 0.62 | 0.63 |
| task|length=mean|h=20 | 4 | 1.03 | 1.02 | 0.25 | 0.25 |
| task|length=pred|sigma=0.5|h=20 | 1.6 | 1.00 | 1.32 | 0.30 | 0.83 |
| task|length=pred|sigma=0.5|h=20 | 4 | 1.02 | 2.12 | 0.03 | 0.52 |
| task|length=oracle|h=20 | 1.6 | 1.00 | 1.36 | 0.19 | 0.85 |
| task|length=oracle|h=20 | 4 | 1.03 | 2.28 | 0.00 | 0.57 |

## Recovery after a surge: fraction of tasks arriving after the surge that complete (median)

| policy | 3.0 | 5.0 |
|:---|:---:|:---:|
| none | 0.55 | 0.60 |
| prog | 1.00 | 0.94 |
| shed|q=16 | 1.00 | 1.00 |
| cap|n=64 | 1.00 | 1.00 |
| task|length=mean|h=20 | 1.00 | 1.00 |
| little|d=5 | 1.00 | 1.00 |
| little|d=5|ident=infer|p_id=0.7 | 1.00 | 1.00 |
| cap|n=128 | 1.00 | 1.00 |
| cap|n=16 | 0.55 | 0.54 |
| cap|n=32 | 0.96 | 0.96 |
| little|d=1 | 1.00 | 1.00 |
| little|d=10 | 1.00 | 1.00 |
| little|d=2.5 | 1.00 | 1.00 |
| little|d=5|fifo=1 | 1.00 | 1.00 |
| little|d=5|hint=1 | 1.00 | 1.00 |
| little|d=5|ident=infer|p_id=0.5 | 1.00 | 1.00 |
| little|d=5|ident=infer|p_id=0.9 | 1.00 | 1.00 |
| shed|q=4 | 1.00 | 1.00 |
| shed|q=64 | 0.75 | 0.76 |
| task|length=mean|h=10 | 0.92 | 0.92 |
| task|length=mean|h=40 | 1.00 | 1.00 |
| task|length=mean|h=5 | 0.48 | 0.48 |
| task|length=mean|h=80 | 1.00 | 1.00 |
| task|length=oracle|h=10 | 0.86 | 0.87 |
| task|length=oracle|h=20 | 0.97 | 0.97 |
| task|length=oracle|h=40 | 1.00 | 1.00 |
| task|length=pred|sigma=0.5|h=10 | 0.84 | 0.85 |
| task|length=pred|sigma=0.5|h=20 | 0.94 | 0.95 |
| task|length=pred|sigma=0.5|h=40 | 0.99 | 0.99 |

## Goodput relative to the best policy per regime, 81 regimes at 2.4x overload (parameters chosen on seeds 0-4, evaluation on seeds 5-9)

| policy | median | 10th pct | worst | regimes within 0.9 of best |
|:---|:---:|:---:|:---:|:---:|
| no admission control | 0.15 | 0.07 | 0.04 | 0% |
| serve by progress | 0.19 | 0.04 | 0.03 | 2% |
| shed (tuned per regime) | 0.68 | 0.57 | 0.47 | 11% |
| shed (one value for all regimes) | 0.65 | 0.52 | 0.47 | 9% |
| cap (tuned per regime) | 0.99 | 0.91 | 0.75 | 91% |
| cap (one value for all regimes) | 0.94 | 0.62 | 0.41 | 65% |
| task work budget (tuned per regime) | 0.99 | 0.88 | 0.56 | 85% |
| task work budget (one value for all regimes) | 0.94 | 0.49 | 0.34 | 59% |
| Little's-law cap (D=5, never tuned) | 0.99 | 0.58 | 0.35 | 85% |

- best shed value per regime: 64: 42, 4: 29, 16: 10

- best cap value per regime: 64: 29, 32: 23, 128: 16, 16: 13

- best task work budget value per regime: 20: 27, 40: 25, 10: 21, 5: 5, 80: 3


KV-bound regimes (kv_tasks=16) cap every policy; ratios are relative to the best policy, not to capacity.

## Retry sensitivity, none, load 2.4: goodput by retry budget (rows) and backoff base (cols)

| retries | 0.5 | 2.0 | 8.0 |
|:---|:---:|:---:|:---:|
| 0 | 0.34 | 0.34 | 0.34 |
| 1 | 0.19 | 0.23 | 0.36 |
| 3 | 0.25 | 0.26 | 0.08 |
| 8 | 0.01 | 0.00 | 0.00 |

## Retry sensitivity, little|d=5, load 2.4: goodput by retry budget (rows) and backoff base (cols)

| retries | 0.5 | 2.0 | 8.0 |
|:---|:---:|:---:|:---:|
| 0 | 1.01 | 1.01 | 1.01 |
| 1 | 1.01 | 1.02 | 1.01 |
| 3 | 1.01 | 1.03 | 1.08 |
| 8 | 1.21 | 1.35 | 1.35 |

## Closed-network wait: measured vs. fluid prediction N s/C - s - tau

| tool s | N | measured | fluid |
|:---:|:---:|:---:|:---:|
| 1 | 16 | 0.00 | 0.00 |
| 1 | 32 | 1.11 | 1.00 |
| 1 | 48 | 3.09 | 3.00 |
| 1 | 64 | 4.99 | 5.00 |
| 1 | 96 | 8.90 | 9.00 |
| 1 | 128 | 12.84 | 13.00 |
| 3 | 16 | 0.00 | 0.00 |
| 3 | 32 | 0.07 | 0.00 |
| 3 | 48 | 1.41 | 1.00 |
| 3 | 64 | 3.35 | 3.00 |
| 3 | 96 | 7.21 | 7.00 |
| 3 | 128 | 11.04 | 11.00 |
| 10 | 16 | 0.00 | 0.00 |
| 10 | 32 | 0.00 | 0.00 |
| 10 | 48 | 0.00 | 0.00 |
| 10 | 64 | 0.03 | 0.00 |
| 10 | 96 | 1.67 | 0.00 |
| 10 | 128 | 5.04 | 4.00 |

Largest absolute deviation: 1.67 s.
