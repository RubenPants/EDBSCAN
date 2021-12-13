# EDBSCAN

Enforced Density-Based Spatial Clustering of Applications with Noise.

This package is an extension on the [DBSCAN](https://arxiv.org/abs/1706.03113) algorithm to enable for pre-labeled data points in the clusters, or in other words to enforce certain cluster values and splits.
It mimics the [scikit-learn](https://scikit-learn.org/stable/) implementation of [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).


## Installation
You can either install the package through [PyPI](https://pypi.org/project/edbscan/):
```shell
pip install edbscan
```

Or via this repository directly:
```shell
pip install git+https://github.com/RubenPants/EDBSCAN.git
```


## Usage

The image below shows you the result of EDBSCAN on a given input. The image on the left shows you the raw input data, together with the few labeled samples. The image on the right shows the clusters found by EDBSCAN, where the light blue dots represent the detected noise.

![Result of EDBSCAN](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/usage.png?raw=true)

```python
# Load in the data
import numpy as np
data = np.load(open('data.npy'))
print(data.shape)  # (220, 2)
y = np.load(open('y.npy'))
print(y.shape)  # (220, )
print(y)  # array([None, None, …, -1, None, …, 0, None, …, 1, …], dtype=object)

# Run the algorithm
from edbscan import edbscan
core_points, labels = edbscan(X=data, y=y)
print(labels)  # array([-1, 2, 2, 4, -1, -1, 6, 3, 4, …])
```

As shown in the code snippet above, aside from the raw data (`data`), a target vector `y` is provided. This vector indicates the known (labeled) clusters. A `None` cluster label are those not yet known, that need to get clustered by the EDBSCAN algorithm.

For more detailed usages, see the notebooks present in the `examples/` folder.


## How EDBSCAN works

There are three concepts that define how EDBSCAN operates:
* The **DBSCAN** algorithm on which this algorithm is based on, read the [paper](https://arxiv.org/abs/1706.03113) or the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) for more.
* **Semi-supervised annotations**, represented by the `y` vector in the *Usage* section. This vector contains three types of values:
  * `None` if the given sample is not known to belong to a specific cluster and needs to get labeled by the EDBSCAN algorithm
  * `-1` if the given sample is known to be noise
  * `0..N` if the given sample is known to belong to cluster `0..N`
* Where DBSCAN expands its clusters in a [FIFO](https://en.wikipedia.org/wiki/FIFO_(computing_and_electronics)) fashion, will EDBSCAN expand its clusters in a **most dense first** fashion. In other words, the items that have the most detected nearest neighbours get expanded first. By doing so, the denser areas get assigned a cluster faster. This prevents two dense cluster that are near each other from merging if they are already assigned a different label.


## Comparison

This section compares EDBSCAN to (1) other clustering algorithms as DBSCAN and HDBSCAN, and (2) on different clustering benchmarks.

### 1. DBSCAN, HDBSCAN, and EDBSCAN

This section compares the behaviour of the [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) algorithm, the [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html) and the [EDBSCAN](https://github.com/RubenPants/EDBSCAN) algorithm on the data shown in the *Usage* section. The input data looks as follows:

![Comparison between DBSCAN, HDBSCAN, and EDBSCAN](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/comparison.png?raw=true)

In each of the clustered results, light-blue data represents the detected noise.

Some observations on the [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) result:
* Green combined two clusters that should be separated
* Purple combined two clusters that should be separated
* Brown identified noise as a cluster

Some observations on the [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html)  result:
* Yellow and Grey are now successfully separated
* Brown and Pink are now successfully separated
* Purple identified noise as a cluster

Some observations on the [EDBSCAN](https://github.com/RubenPants/EDBSCAN) result:
* Grey and Orange are now successfully separated
* Brown and Pink are now successfully separated
* The noise that was previously detected as a cluster is now successfully identified as noise


### 2. Scikit-learn cluster benchmark

The following images show the results of the EDBSCAN algorithm on different [scikit-learn clustering benchmarks](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py).

![circles](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/circles.png?raw=true)

![moons](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/moons.png?raw=true)

![blobs](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/blobs.png?raw=true)

![aniso](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/aniso.png?raw=true)

![uniform](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/uniform.png?raw=true)

![multi](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/multi.png?raw=true)
