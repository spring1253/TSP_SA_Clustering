# Corrected experiment protocol

This branch keeps the original research idea while repairing the experimental
errors found in the bachelor-thesis implementation.

## Corrections

1. Methods share the same explicitly seeded instances. There is no implicit
   `random.shuffle` in the data loader.
2. Compute timing excludes plotting and result-file I/O.
3. SA uses a real, fixed proposal budget and a scale-calibrated temperature.
4. The SA neighborhood uses O(1) 2-opt cost deltas instead of recomputing the
   whole tour for every proposal.
5. A source-data tour is treated as a `reference`, not an `optimum`, because the
   source generator used Concorde's GEO norm while this project evaluates plain
   Euclidean distance. Exact ratios are emitted only when Held-Karp solves the
   same distance matrix.

Every output row validates that the returned route visits every vertex exactly once.

## Reproduce the corrected pilot

```bash
python simulation.py \
  --sizes 10 100 300 1000 \
  --seeds 101 202 303 \
  --budgets 2000 8000 32000 \
  --methods hierarchical \
  --output experiment_results_v2/corrected_pilot.csv
```

The `iterations` column is the total number of 2-opt proposals. Clustered SA
splits that same total budget between the centroid tour and all local tours. A
full Pareto study must additionally compare equal wall-clock budgets.

The paired replacement for the original 1,000-node clustering-method experiment is:

```bash
python simulation.py \
  --sizes 1000 \
  --seeds 101 202 303 404 505 606 707 808 \
  --budgets 32000 \
  --methods greedy kmeans hierarchical spectral \
  --cluster-counts 16 22 32 44 64 \
  --output experiment_results_v2/clustering_methods_n1000.csv
```
