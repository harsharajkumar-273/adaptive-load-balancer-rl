# Trust gate: regret of the model-driven router against a non-learned fallback, paired by seed

Regret = (P99 of policy - P99 of fallback) / P99 of fallback, per seed and cell (16 routers, 10 seeds). `mean`, `p90` (worst decile) and the share of runs worse than the fallback are over all seeds and the 8 cells (steady/gray x 50/200 ms x synchronised/not) unless a cell is named.


## Fallback: P2C

| model | policy | mean | p90 | worse than fallback | worse by >2x |
|:---|:---|:---:|:---:|:---:|:---:|
| linear model | ungated | +6.93 | +20.59 | 65% | 50% |
| linear model | hard gate -> P2C | +2.01 | +4.16 | 24% | 12% |
| linear model | hard gate -> JSQ | +7.23 | +14.85 | 80% | 46% |
| linear model | mix p=.25 -> P2C | -0.18 | -0.10 | 1% | 0% |
| linear model | mix p=.5 -> P2C | +0.08 | +0.06 | 12% | 5% |
| linear model | mix p=.75 -> P2C | +0.60 | +1.92 | 31% | 16% |
| linear model | learned P2C (no gate) | -0.43 | -0.29 | 2% | 0% |
| quadratic model (confidently wrong) | ungated | +167.67 | +340.12 | 100% | 99% |
| quadratic model (confidently wrong) | hard gate -> P2C | +99.45 | +318.21 | 84% | 79% |
| quadratic model (confidently wrong) | mix p=.5 -> P2C | +4.04 | +19.43 | 45% | 26% |
| ewma model (mediocre) | ungated | +21.69 | +56.52 | 68% | 61% |
| ewma model (mediocre) | hard gate -> P2C | +1.43 | +0.18 | 14% | 8% |

## Fallback: JSQ

| model | policy | mean | p90 | worse than fallback | worse by >2x |
|:---|:---|:---:|:---:|:---:|:---:|
| linear model | ungated | +2.51 | +8.31 | 50% | 38% |
| linear model | hard gate -> P2C | -0.02 | +0.10 | 11% | 9% |
| linear model | hard gate -> JSQ | +2.40 | +6.78 | 51% | 25% |
| linear model | mix p=.25 -> P2C | -0.52 | -0.15 | 0% | 0% |
| linear model | mix p=.5 -> P2C | -0.49 | -0.23 | 5% | 0% |
| linear model | mix p=.75 -> P2C | -0.29 | -0.14 | 6% | 4% |
| linear model | learned P2C (no gate) | -0.66 | -0.36 | 0% | 0% |
| quadratic model (confidently wrong) | ungated | +107.09 | +250.99 | 100% | 99% |
| quadratic model (confidently wrong) | hard gate -> P2C | +55.85 | +157.67 | 80% | 76% |
| quadratic model (confidently wrong) | mix p=.5 -> P2C | +1.78 | +6.48 | 29% | 18% |
| ewma model (mediocre) | ungated | +9.21 | +24.01 | 65% | 59% |
| ewma model (mediocre) | hard gate -> P2C | -0.19 | -0.20 | 6% | 5% |

## Per cell (fallback P2C): mean regret [p90]

| cell | ungated | hard gate | mix .5 | ungated q2 | gated q2 | mix .5 q2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| steady 50 ms sync | +6.41 [+18.35] | -0.21 [-0.17] | -0.27 [-0.23] | +215.53 [+364.04] | +27.74 [+68.40] | +11.00 [+28.32] |
| steady 50 ms unsync | +4.07 [+18.22] | -0.29 [-0.22] | -0.31 [-0.26] | +235.61 [+447.71] | +108.16 [+237.16] | +4.09 [+4.20] |
| steady 200 ms sync | +21.51 [+37.84] | +1.81 [+4.82] | +0.57 [+1.01] | +141.75 [+221.32] | +114.99 [+233.44] | +2.80 [+2.89] |
| steady 200 ms unsync | +15.63 [+31.69] | -0.10 [-0.03] | -0.24 [-0.18] | +179.66 [+263.20] | +307.73 [+545.13] | +12.08 [+26.67] |
| gray 50 ms sync | +0.96 [+2.96] | -0.21 [-0.17] | -0.29 [-0.22] | +176.90 [+367.21] | +71.28 [+211.37] | +0.06 [+0.51] |
| gray 50 ms unsync | -0.31 [-0.23] | -0.32 [-0.27] | -0.34 [-0.30] | +171.05 [+316.48] | +6.10 [+19.13] | +0.30 [+1.43] |
| gray 200 ms sync | +4.59 [+10.02] | +14.86 [+39.88] | +1.65 [+4.31] | +78.95 [+159.25] | +66.18 [+166.79] | +1.53 [+3.69] |
| gray 200 ms unsync | +2.56 [+3.87] | +0.55 [+0.77] | -0.17 [-0.07] | +141.94 [+246.40] | +93.40 [+202.15] | +0.46 [+1.10] |

## Window sensitivity (hard gate to P2C, synchronised refresh): mean regret / p90

                         mean  <lambda_0>
scenario staleness w                     
gray     0.05      10   -0.13        0.03
                   25    2.19        2.44
                   50   -0.21       -0.17
                   100  -0.16       -0.08
                   200  -0.22       -0.15
         0.20      10    9.54       16.22
                   25   16.19       44.99
                   50   14.86       39.88
                   100  10.89       37.39
                   200   4.29       11.94
steady   0.05      10   -0.11       -0.02
                   25   -0.21       -0.17
                   50   -0.21       -0.17
                   100  -0.23       -0.17
                   200  -0.26       -0.21
         0.20      10    0.29        0.35
                   25    7.17       19.87
                   50    1.81        4.82
                   100   0.34        1.15
                   200  -0.12       -0.04