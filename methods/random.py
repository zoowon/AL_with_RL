import random

from typing import Tuple, List


def random_sampling(labeled_indices: List[int], unlabeled_indices: List[int], addendum: int):
    if len(unlabeled_indices) == 0:
        return labeled_indices, unlabeled_indices

    add_count = min(addendum, len(unlabeled_indices))
    newly_labeled = random.sample(unlabeled_indices, add_count)

    new_labeled = labeled_indices + newly_labeled
    remaining_unlabeled = [idx for idx in unlabeled_indices if idx not in newly_labeled]

    return new_labeled, remaining_unlabeled