# Corrected pilot results

Generated on 2026-09-18 with the commands documented in `EXPERIMENTS_V2.md`.
All methods saw the same seeded uniform Euclidean instances. Every returned tour
passed the permutation validity check. Times exclude plotting and result-file I/O.

## Global SA vs hierarchical clustered SA

Means over seeds 101, 202, and 303. Both algorithms received the same total
number of 2-opt proposals.

| n | proposals | global length | clustered length | clustered/global | global seconds | clustered seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2,000 | 2.501 | 2.606 | 1.043 | 0.0066 | 0.0146 |
| 100 | 2,000 | 17.160 | 9.517 | 0.555 | 0.0060 | 0.0114 |
| 100 | 32,000 | 9.245 | 8.851 | 0.958 | 0.0870 | 0.1011 |
| 300 | 32,000 | 26.948 | 15.640 | 0.581 | 0.1174 | 0.1064 |
| 1,000 | 2,000 | 356.754 | 67.461 | 0.189 | 0.0171 | 0.0648 |
| 1,000 | 32,000 | 132.125 | 33.701 | 0.255 | 0.2158 | 0.1667 |

For n=10, Held-Karp computed the exact Euclidean optimum. Global SA reached it
on all three instances, while clustered SA was 4.3%-5.6% above optimum on
average. For larger n, no approximation ratio is claimed.

The clustered method is much better than random-start global SA at larger n
under this limited proposal budget. This is promising but is not yet a research
conclusion: global SA is a deliberately simple baseline and must be strengthened
with nearest-neighbor initialization, candidate lists, and multi-start runs.

## Paired clustering-method grid at n=1,000

Means over eight common instances with 32,000 proposals per run.

| method | k=16 | k=22 | k=32 | k=44 | k=64 |
|---|---:|---:|---:|---:|---:|
| Greedy | 43.940 | 39.409 | 35.785 | 34.055 | 33.196 |
| Hierarchical | 41.409 | 37.121 | 33.713 | 31.802 | **31.496** |
| K-Means | **40.731** | **36.952** | 33.797 | 32.121 | 31.764 |
| Spectral | 41.551 | 37.368 | 34.560 | 32.520 | 31.860 |

Unlike the thesis experiment, all four methods used the same eight instances.
K-Means was marginally best for k=16 and k=22; Hierarchical was best for k=32,
k=44, and k=64. With a fixed total proposal budget, increasing k improved tour
length in this pilot, the opposite of the old result. Smaller local problems let
the fixed budget search each cluster more effectively; this interaction must be
separated from the intrinsic effect of clustering in the full study.

## Interpretation limits

These files are a smoke-test-quality corrected pilot, not final benchmark data.
They cover one distribution, few instances, one machine, and a weak global-SA
baseline. A full Pareto study needs equal wall-clock budgets, stronger classical
baselines, confidence intervals, multiple spatial distributions, and TSPLIB.
