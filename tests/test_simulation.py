import unittest

import numpy as np

from simulation import (
    clustered_sa_tsp, distance_matrix, generate_uniform_instance,
    held_karp_length, simulated_annealing_tsp, tour_length, validate_tour,
)


class SimulationTests(unittest.TestCase):
    def test_tour_length_is_symmetric(self):
        vertices = generate_uniform_instance(4, 7)
        distances = distance_matrix(vertices)
        route = [0, 1, 2, 3]
        self.assertEqual(tour_length(route, distances), tour_length(route[::-1], distances))

    def test_sa_is_reproducible_and_valid(self):
        vertices = generate_uniform_instance(30, 11)
        distances = distance_matrix(vertices)
        first, first_cost = simulated_annealing_tsp(range(30), distances, 2_000, 99)
        second, second_cost = simulated_annealing_tsp(range(30), distances, 2_000, 99)
        validate_tour(first, 30)
        self.assertEqual(first, second)
        self.assertEqual(first_cost, second_cost)

    def test_clustered_sa_is_valid(self):
        vertices = generate_uniform_instance(50, 13)
        route, cost = clustered_sa_tsp(vertices, 7, "hierarchical", 3_000, 123)
        validate_tour(route, 50)
        self.assertGreater(cost, 0)

    def test_held_karp_triangle(self):
        distances = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertEqual(held_karp_length(distances), 3.0)


if __name__ == "__main__":
    unittest.main()
