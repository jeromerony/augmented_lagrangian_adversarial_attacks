from typing import Tuple

from torch import Tensor


def robust_accuracy_curve(distances: Tensor,
                          successes: Tensor,
                          worst_distance: float = float('inf')) -> Tuple[Tensor, Tensor]:
    worst_case_distances = distances.clone()
    worst_case_distances[~successes] = worst_distance
    unique_distances = worst_case_distances.unique()
    robust_accuracies = (worst_case_distances.unsqueeze(0) > unique_distances.unsqueeze(1)).float().mean(1)
    return unique_distances, robust_accuracies
