"""Python version of the EDBSCAN inner loop."""

from logging import warning
from typing import List

import numpy as np
from numpy.typing import NDArray


def pop(
    stack: List[int],
    neighborhoods: NDArray[np.int8],
) -> int:
    """Pop index from stack with the highest density, calculated as number of neighbors."""
    best_size, best_idx = -1, -1
    for idx in stack:
        if len(neighborhoods[idx]) > best_size:
            best_size = len(neighborhoods[idx])
            best_idx = idx

    # Pop the best index from the stack.
    stack.remove(best_idx)
    return best_idx


def get_unlabeled(
    is_core: NDArray[np.int8],
    neighborhoods: NDArray[np.int8],
    labels: NDArray[np.int8],
) -> int:
    """Get the best unlabeled core index."""
    best_size, best_idx = -1, -1
    for idx in range(neighborhoods.shape[0]):
        if not is_core[idx]:
            continue
        if labels[idx] != -2:
            continue
        if len(neighborhoods[idx]) > best_size:
            best_size = len(neighborhoods[idx])
            best_idx = idx

    # Pop the best index from the stack.
    return best_idx


def conflict(
    neighborhood: NDArray[np.int8],
    labels: NDArray[np.int8],
) -> bool:
    """Check if there's a conflict in the given neighborhood."""
    neighborhood_labels = {label for label in labels[neighborhood] if label != -2}

    # Noise always leads to a conflict.
    if -1 in neighborhood_labels:
        return True

    # If more than one cluster label is present, there's a conflict.
    return len(neighborhood_labels - {-1}) > 1


def inner(
    is_core: NDArray[np.int8],
    neighborhoods: NDArray[np.int8],
    labels: NDArray[np.int8],
) -> None:
    """Inner-loop of the EDBSCAN algorithm."""
    warning("RUNNING PYTHON LOOP, PLEASE USE C++ OPTIMISED LOOP INSTEAD")

    # Initialise the stack with all known clusters.
    stack = []
    for idx in range(labels.shape[0]):
        if labels[idx] >= 0:
            stack.append(idx)

    # Density-first search, where the density is defined by the number of neighbors a component has.
    # The algorithm expands the most dense core components first and ends at the non-core points.
    # Non-core points are labeled as part of a component, but don't expand their neighborhoods.
    while True:
        if len(stack) == 0:
            idx = get_unlabeled(is_core=is_core, neighborhoods=neighborhoods, labels=labels)

            # Break if no unlabeled found
            if idx == -1:
                break

            # Break if conflict found
            neighborhood = neighborhoods[idx]
            if conflict(neighborhood=neighborhood, labels=labels):
                break

            # Add to stack (new cluster found that doesn't lead to a conflict)
            labels[idx] = max(max(labels) + 1, 0)
            stack.append(idx)

        # Pop the most dense index in stack and expand if there's no conflict.
        # Only add unlabeled core indices to the stack.
        idx = pop(stack=stack, neighborhoods=neighborhoods)
        if not conflict(neighborhood=neighborhoods[idx], labels=labels):
            for n_idx in neighborhoods[idx]:
                if labels[n_idx] == -2 and is_core[n_idx]:
                    labels[n_idx] = labels[idx]
                    stack.append(n_idx)
