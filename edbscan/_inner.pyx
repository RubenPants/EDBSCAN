# Fast inner loop for EDBSCAN
# Author: Ruben Broekx
#
# Code forked from Scikit-Learn's DBSCAN
# Author: Lars Buitinck
# License: 3-clause BSD
#
# cython: boundscheck=False, wraparound=False, infer_types=True

cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

np.import_array()


cdef int pop(
    list stack,
    np.ndarray[object, ndim=1] neighborhoods,
): 
    """Pop index from stack with the highest density, calculated as number of neighbors."""
    cdef int best_size = -1
    cdef int best_idx = -1
    cdef int idx
    cdef np.ndarray[np.npy_intp, ndim=1, mode='c'] neighborhood

    for idx in stack:
        neighborhood = neighborhoods[idx]
        if neighborhood.shape[0] > best_size:
            best_size = neighborhood.shape[0]
            best_idx = idx
    
    # Pop the best index from the stack.
    stack.remove(best_idx)
    return best_idx


cdef int get_unlabeled(
    np.ndarray[np.uint8_t, ndim=1, mode='c'] is_core,
    np.ndarray[object, ndim=1] neighborhoods,
    np.ndarray[np.npy_intp, ndim=1, mode='c'] labels,
):
    """Get the best unlabeled core index."""
    cdef int best_size = -1
    cdef int best_idx = -1
    cdef np.ndarray[np.npy_intp, ndim=1, mode='c'] neighborhood

    for idx in xrange(neighborhoods.shape[0]):
        if not is_core[idx]:
            continue
        if labels[idx] != -2:
            continue

        neighborhood = neighborhoods[idx]
        if neighborhood.shape[0] > best_size:
            best_size = neighborhood.shape[0]
            best_idx = idx
    
    # Pop the best index from the stack.
    return best_idx


cdef bool value_in_array(
    int val,
    vector[np.npy_intp] arr,
):
    """Check if the value exists in the array."""
    cdef int i

    for i in arr:
        if i == val:
            return True
    return False


cdef int get_max_0(
    np.ndarray[np.npy_intp, ndim=1, mode='c'] arr,
):
    """Get the maximum value in the array, that is at least zero."""
    cdef int max_val = 0
    cdef int i, idx

    for i in xrange(arr.shape[0]):
        idx = arr[i]
        if i > max_val:
            max_val = i
    return max_val

cdef bool conflict(
    np.ndarray[np.npy_intp, ndim=1] neighborhood,
    np.ndarray[np.npy_intp, ndim=1, mode='c'] labels,
):
    """Check if there's a conflict in the given neighborhood."""
    cdef int label
    cdef vector[np.npy_intp] seen

    for label in labels[neighborhood]:
        if label == -2:
            continue
        if value_in_array(label, seen):
            continue
        if label == -1:
            return True
        seen.push_back(label)
    return seen.size() > 1


cpdef void inner(
    np.ndarray[np.uint8_t, ndim=1, mode='c'] is_core,
    np.ndarray[object, ndim=1] neighborhoods,
    np.ndarray[np.npy_intp, ndim=1, mode='c'] labels,
):
    """Inner-loop of the EDBSCAN algorithm."""
    # Initialise the stack with all known clusters.
    cdef int i, idx, n_idx
    cdef list stack = []
    cdef np.ndarray[np.npy_intp, ndim=1, mode='c'] neighborhood
    
    for idx in xrange(labels.shape[0]):
        if labels[idx] >= 0:
            stack.append(idx)
    
    # Density-first search, where the density is defined by the number of neighbors a component has.
    # The algorithm expands the most dense core components first and ends at the non-core points.
    # Non-core points are labeled as part of a component, but don't expand their neighborhoods.
    while True:
        if len(stack) == 0:
            idx = get_unlabeled(is_core=is_core, neighborhoods=neighborhoods, labels=labels)
            
            # Add to stack if a new cluster is found that does not lead to a conflict.
            neighborhood = neighborhoods[idx]
            if idx >= 0 and not conflict(neighborhood=neighborhood, labels=labels):
                labels[idx] = get_max_0(labels)
                stack.append(idx)
            
            # No unlabeled cluster left -> Break while-loop.
            else:
                break
        
        # Pop the most dense index in stack and expand if there's no conflict.
        # Only add unlabeled core indices to the stack.
        idx = pop(stack=stack, neighborhoods=neighborhoods)
        neighborhood = neighborhoods[idx]
        if not conflict(neighborhood=neighborhood, labels=labels):
            for i in xrange(neighborhood.shape[0]):
                n_idx = neighborhood[i]
                if labels[n_idx] == -2 and is_core[n_idx]:
                    labels[n_idx] = labels[idx]
                    stack.append(n_idx)
