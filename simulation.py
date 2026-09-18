"""Reproducible clustering + simulated annealing experiments for Euclidean TSP.

Every method receives the same seeded instance; plotting and file I/O are outside
the measured region; iterations are a real proposal budget; 2-opt deltas are O(1);
and ratios are reported only against an exact solution under the same metric.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


@dataclass(frozen=True)
class Vertex:
    id: int
    x: float
    y: float

    def distance_to(self, other: "Vertex") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def generate_uniform_instance(num_vertices: int, seed: int) -> list[Vertex]:
    rng = np.random.default_rng(seed)
    points = rng.random((num_vertices, 2))
    return [Vertex(i, float(x), float(y)) for i, (x, y) in enumerate(points)]


def read_data(data_path: str | Path, instance_indices: Sequence[int]) -> list[tuple[list[Vertex], list[int]]]:
    """Read explicitly selected instances without implicit random shuffling.

    The returned tour is a reference tour, not a claimed Euclidean optimum. The
    source generator used Concorde's GEO norm while this project evaluates plain
    Euclidean distance.
    """
    wanted = set(instance_indices)
    found: dict[int, tuple[list[Vertex], list[int]]] = {}
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle):
            if line_number not in wanted:
                continue
            tokens = raw.split()
            marker = tokens.index("output")
            coordinates = [float(value) for value in tokens[:marker]]
            vertices = [
                Vertex(i // 2, coordinates[i], coordinates[i + 1])
                for i in range(0, len(coordinates), 2)
            ]
            reference = [int(value) - 1 for value in tokens[marker + 1 :]]
            if len(reference) == len(vertices) + 1 and reference[0] == reference[-1]:
                reference.pop()
            validate_tour(reference, len(vertices))
            found[line_number] = (vertices, reference)
            if len(found) == len(wanted):
                break
    missing = wanted - found.keys()
    if missing:
        raise IndexError(f"Instance indices not found: {sorted(missing)}")
    return [found[index] for index in instance_indices]


def distance_matrix(vertices: Sequence[Vertex]) -> np.ndarray:
    xy = np.asarray([(vertex.x, vertex.y) for vertex in vertices], dtype=float)
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def tour_length(route: Sequence[int], distances: np.ndarray) -> float:
    if len(route) < 2:
        return 0.0
    order = np.asarray(route, dtype=int)
    return float(distances[order, np.roll(order, -1)].sum())


def validate_tour(route: Sequence[int], num_vertices: int) -> None:
    if len(route) != num_vertices or set(route) != set(range(num_vertices)):
        raise ValueError("Tour must contain every vertex exactly once")


def _sample_two_opt_indices(size: int, rng: random.Random) -> tuple[int, int]:
    while True:
        i, j = sorted(rng.sample(range(size), 2))
        if j - i >= 2 and not (i == 0 and j == size - 1):
            return i, j


def _two_opt_delta(route: Sequence[int], distances: np.ndarray, i: int, j: int) -> float:
    before_i, at_i = route[i - 1], route[i]
    at_j, after_j = route[j], route[(j + 1) % len(route)]
    removed = distances[before_i, at_i] + distances[at_j, after_j]
    added = distances[before_i, at_j] + distances[at_i, after_j]
    return float(added - removed)


def _calibrated_temperature(
    route: Sequence[int], distances: np.ndarray, rng: random.Random, target_acceptance: float = 0.8
) -> float:
    positive_deltas: list[float] = []
    for _ in range(min(128, 4 * len(route))):
        i, j = _sample_two_opt_indices(len(route), rng)
        delta = _two_opt_delta(route, distances, i, j)
        if delta > 0:
            positive_deltas.append(delta)
    if not positive_deltas:
        nonzero = distances[distances > 0]
        return float(np.mean(nonzero)) if nonzero.size else 1.0
    return -float(np.mean(positive_deltas)) / math.log(target_acceptance)


def simulated_annealing_tsp(
    route_vertices: Sequence[int],
    distances: np.ndarray,
    iterations: int,
    seed: int,
    initial_route: Sequence[int] | None = None,
) -> tuple[list[int], float]:
    """Run SA using a fixed proposal budget and O(1) 2-opt cost deltas."""
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if len(route_vertices) < 4 or iterations == 0:
        route = list(initial_route if initial_route is not None else route_vertices)
        return route, tour_length(route, distances)

    rng = random.Random(seed)
    if initial_route is None:
        current = list(route_vertices)
        rng.shuffle(current)
    else:
        current = list(initial_route)
    current_cost = tour_length(current, distances)
    best, best_cost = current.copy(), current_cost

    temperature = max(_calibrated_temperature(current, distances, rng), 1e-12)
    final_temperature = max(temperature * 1e-3, 1e-12)
    cooling_rate = math.exp(math.log(final_temperature / temperature) / max(iterations, 1))

    for _ in range(iterations):
        i, j = _sample_two_opt_indices(len(current), rng)
        delta = _two_opt_delta(current, distances, i, j)
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current[i : j + 1] = reversed(current[i : j + 1])
            current_cost += delta
            if current_cost < best_cost:
                best, best_cost = current.copy(), current_cost
        temperature = max(temperature * cooling_rate, 1e-12)
    return best, tour_length(best, distances)


def _cluster_labels(vertices: Sequence[Vertex], num_clusters: int, method: str, seed: int) -> np.ndarray:
    data = np.asarray([(vertex.x, vertex.y) for vertex in vertices], dtype=float)
    if method == "kmeans":
        return KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit_predict(data)
    if method == "hierarchical":
        return AgglomerativeClustering(n_clusters=num_clusters).fit_predict(data)
    if method == "spectral":
        return SpectralClustering(
            n_clusters=num_clusters, affinity="nearest_neighbors", random_state=seed
        ).fit_predict(data)
    if method == "greedy":
        rng = random.Random(seed)
        centers = [rng.randrange(len(vertices))]
        nearest = np.linalg.norm(data - data[centers[0]], axis=1)
        while len(centers) < num_clusters:
            next_center = int(np.argmax(nearest))
            centers.append(next_center)
            nearest = np.minimum(nearest, np.linalg.norm(data - data[next_center], axis=1))
        center_xy = data[centers]
        return np.argmin(np.linalg.norm(data[:, None, :] - center_xy[None, :, :], axis=2), axis=1)
    raise ValueError(f"Unknown clustering method: {method}")


def _centroids(vertices: Sequence[Vertex], labels: np.ndarray, num_clusters: int) -> np.ndarray:
    data = np.asarray([(vertex.x, vertex.y) for vertex in vertices], dtype=float)
    return np.vstack([data[labels == cluster_id].mean(axis=0) for cluster_id in range(num_clusters)])


def _open_cluster_cycle(
    cycle: Sequence[int],
    prev_centroid: np.ndarray,
    next_centroid: np.ndarray,
    vertices: Sequence[Vertex],
    distances: np.ndarray,
) -> list[int]:
    if len(cycle) <= 1:
        return list(cycle)
    coordinates = np.asarray([(vertex.x, vertex.y) for vertex in vertices], dtype=float)
    best_score, best_path = math.inf, None
    for edge_index in range(len(cycle)):
        u, v = cycle[edge_index], cycle[(edge_index + 1) % len(cycle)]
        forward = list(cycle[edge_index + 1 :]) + list(cycle[: edge_index + 1])
        for candidate in (forward, list(reversed(forward))):
            score = (
                np.linalg.norm(coordinates[candidate[0]] - prev_centroid)
                + np.linalg.norm(coordinates[candidate[-1]] - next_centroid)
                - distances[u, v]
            )
            if score < best_score:
                best_score, best_path = float(score), candidate
    assert best_path is not None
    return best_path


def clustered_sa_tsp(
    vertices: Sequence[Vertex], num_clusters: int, method: str, iterations: int, seed: int,
    distances: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Cluster-first SA with one shared, explicit total proposal budget."""
    if not 1 < num_clusters < len(vertices):
        raise ValueError("num_clusters must be between 2 and n-1")
    labels = _cluster_labels(vertices, num_clusters, method, seed)
    centers = _centroids(vertices, labels, num_clusters)
    center_distances = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2))
    center_budget = min(max(num_clusters * 10, iterations // 10), iterations)
    local_budget = iterations - center_budget
    cluster_order, _ = simulated_annealing_tsp(
        list(range(num_clusters)), center_distances, center_budget, seed + 1
    )

    if distances is None:
        distances = distance_matrix(vertices)
    cluster_members = {cid: np.flatnonzero(labels == cid).tolist() for cid in range(num_clusters)}
    cycles: dict[int, list[int]] = {}
    allocated = 0
    for position, cid in enumerate(cluster_order):
        members = cluster_members[cid]
        if position == len(cluster_order) - 1:
            budget = local_budget - allocated
        else:
            budget = int(local_budget * len(members) / len(vertices))
            allocated += budget
        cycles[cid], _ = simulated_annealing_tsp(members, distances, max(budget, 0), seed + 1000 + cid)

    merged: list[int] = []
    for position, cid in enumerate(cluster_order):
        prev_center = centers[cluster_order[position - 1]]
        next_center = centers[cluster_order[(position + 1) % len(cluster_order)]]
        merged.extend(_open_cluster_cycle(cycles[cid], prev_center, next_center, vertices, distances))
    validate_tour(merged, len(vertices))
    return merged, tour_length(merged, distances)


def held_karp_length(distances: np.ndarray) -> float:
    """Exact TSP length for small instances under this exact distance matrix."""
    n = len(distances)
    if n > 18:
        raise ValueError("Held-Karp is intentionally limited to n <= 18")
    costs = {(1 << k, k): float(distances[0, k]) for k in range(1, n)}
    for subset_size in range(2, n):
        next_costs: dict[tuple[int, int], float] = {}
        for mask in range(1, 1 << (n - 1)):
            if mask.bit_count() != subset_size:
                continue
            actual_mask = mask << 1
            for last in range(1, n):
                last_bit = 1 << last
                if not actual_mask & last_bit:
                    continue
                prev_mask = actual_mask ^ last_bit
                next_costs[(actual_mask, last)] = min(
                    costs[(prev_mask, prev)] + float(distances[prev, last])
                    for prev in range(1, n)
                    if prev_mask & (1 << prev)
                )
        costs = next_costs
    full_mask = ((1 << n) - 1) ^ 1
    return min(costs[(full_mask, last)] + float(distances[last, 0]) for last in range(1, n))


RESULT_FIELDS = [
    "instance_id", "n", "algorithm", "clustering_method", "num_clusters",
    "iterations", "seed", "reference_type", "reference_length",
    "solution_length", "relative_to_reference", "compute_seconds", "valid_tour",
]


def run_experiment(
    sizes: Iterable[int], instance_seeds: Sequence[int], budgets: Sequence[int],
    methods: Sequence[str], output_path: str | Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in sizes:
        k = max(2, round(math.sqrt(n)))
        for instance_id, instance_seed in enumerate(instance_seeds):
            vertices = generate_uniform_instance(n, instance_seed)
            distances = distance_matrix(vertices)
            exact = held_karp_length(distances) if n <= 18 else None
            for budget in budgets:
                algorithms = [("global_sa", None)] + [("clustered_sa", method) for method in methods]
                for algorithm, method in algorithms:
                    run_seed = instance_seed * 100_000 + budget
                    started = time.perf_counter()
                    if algorithm == "global_sa":
                        route, length = simulated_annealing_tsp(list(range(n)), distances, budget, run_seed)
                    else:
                        assert method is not None
                        route, length = clustered_sa_tsp(
                            vertices, k, method, budget, run_seed, distances=distances
                        )
                    elapsed = time.perf_counter() - started
                    validate_tour(route, n)
                    rows.append({
                        "instance_id": instance_id, "n": n, "algorithm": algorithm,
                        "clustering_method": method or "", "num_clusters": k if method else "",
                        "iterations": budget, "seed": instance_seed,
                        "reference_type": "exact_euclidean" if exact is not None else "",
                        "reference_length": exact if exact is not None else "",
                        "solution_length": length,
                        "relative_to_reference": length / exact if exact is not None else "",
                        "compute_seconds": elapsed, "valid_tour": True,
                    })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_cluster_grid(
    n: int,
    instance_seeds: Sequence[int],
    budget: int,
    methods: Sequence[str],
    cluster_counts: Sequence[int],
    output_path: str | Path,
) -> list[dict[str, object]]:
    """Paired replacement for the thesis clustering-method experiment."""
    rows: list[dict[str, object]] = []
    for instance_id, instance_seed in enumerate(instance_seeds):
        vertices = generate_uniform_instance(n, instance_seed)
        distances = distance_matrix(vertices)
        exact = held_karp_length(distances) if n <= 18 else None
        for num_clusters in cluster_counts:
            for method_index, method in enumerate(methods):
                run_seed = instance_seed * 100_000 + budget + method_index
                started = time.perf_counter()
                route, length = clustered_sa_tsp(
                    vertices, num_clusters, method, budget, run_seed, distances=distances
                )
                elapsed = time.perf_counter() - started
                validate_tour(route, n)
                rows.append({
                    "instance_id": instance_id, "n": n, "algorithm": "clustered_sa",
                    "clustering_method": method, "num_clusters": num_clusters,
                    "iterations": budget, "seed": instance_seed,
                    "reference_type": "exact_euclidean" if exact is not None else "",
                    "reference_length": exact if exact is not None else "",
                    "solution_length": length,
                    "relative_to_reference": length / exact if exact is not None else "",
                    "compute_seconds": elapsed, "valid_tour": True,
                })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 100, 300, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2_000, 8_000, 32_000])
    parser.add_argument("--methods", nargs="+", choices=["greedy", "kmeans", "hierarchical", "spectral"], default=["hierarchical"])
    parser.add_argument("--cluster-counts", type=int, nargs="+")
    parser.add_argument("--output", default="experiment_results_v2/corrected_pilot.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cluster_counts:
        if len(args.sizes) != 1 or len(args.budgets) != 1:
            raise SystemExit("--cluster-counts requires exactly one --sizes value and one --budgets value")
        run_cluster_grid(
            args.sizes[0], args.seeds, args.budgets[0], args.methods,
            args.cluster_counts, args.output,
        )
    else:
        run_experiment(args.sizes, args.seeds, args.budgets, args.methods, args.output)
