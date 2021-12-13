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
data = ...
print(data.shape)  # (220, 2)
y = ...
print(y.shape)  # (220, )
print(y)  # array([None, None, ..., -1, None, ..., 0, None, ..., 1, ...], dtype=object)

# Run the algorithm
from edbscan import edbscan
core_points, labels = edbscan(X=data, y=y)
print(labels)  # array([-1, 2, 2, 4, -1, -1, 6, 3, 4, ...])
```

As shown in the code snippet above, aside from the raw data (`data`), a target vector `y` is provided. This vector indicates the known (labeled) clusters. A `None` cluster label are those not yet known, that need to get clustered by the EDBSCAN algorithm.


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

![Input data](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/default_labels.png?raw=true)

#### DBSCAN

The image below shows the clustered result after running the [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) algorithm. The light-blue data points are considered noise. Some observations on the result:
* Green (1) combined two clusters that should be separated
* Purple (3) combined two clusters that should be separated
* Brown (4) identified noise as a cluster

![DBSCAN result](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/default_dbscan.png?raw=true)

#### HDBSCAN

The image below shows the clustered result after running the [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html) algorithm. The light-blue data points are considered noise. Some observations on the result:
* Green (1) and Pink (5) are now successfully separated
* Grey (6) and Yellow (7) are now successfully separated
* Purple (3) identified noise as a cluster

![HDBSCAN result](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/default_hdbscan.png?raw=true)

#### EDBSCAN

The image below shows the clustered result after running the [EDBSCAN](https://github.com/RubenPants/EDBSCAN) algorithm. The light-blue data points are considered noise. Some observations on the result:
* Orange (0) and Grey (6) are now successfully separated
* Brown (4) and Pink (5) are now successfully separated
* The noise that was previously detected as a cluster is now successfully identified as noise

![EDBSCAN result](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/default_edbscan.png?raw=true)

### 2. Scikit-learn cluster benchmark

The following images show the results of the EDBSCAN algorithm on different [scikit-learn clustering benchmarks](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py).

![circles](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/circles.png?raw=true)

![moons](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/moons.png?raw=true)

![blobs](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/blobs.png?raw=true)

![aniso](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/aniso.png?raw=true)

![uniform](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/uniform.png?raw=true)

![multi](https://github.com/RubenPants/EDBSCAN/blob/master/examples/images/multi.png?raw=true)
