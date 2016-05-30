mcores
======
clustering with modal-set estimation


Usage
======

API is modelled after that of sklearn's clustering algorithms.

**Initializiation**:

.. code-block:: python

  MCores(k, beta, epsilon=0, cluster_threshold=None) 
  
k: number of neighbors in k-NN

beta: ranges between 0 and 1. When the procedure is at level F, it examines level-set of kNN graph at level F - beta * F - epsilon

epsilon: used for pruning away false cluster-cores

cluster_threshold: Determines the minimum threshold distance to classify points to its corresponding closest (Hausdorff distance) estimated modal-set. All other points do not get assigned to a cluster. If this paramter is None, then all points get assigned to some cluster.

**Finding Clusters**:

.. code-block:: python

  fit(X)
  predict(X)
  
X is the data matrix, where each row is a datapoint in the euclidean space


**Example** (mixture of two gaussians):

.. code-block:: python

  from MCores import *
  import numpy as np
  
  a = [np.random.normal(0, 1, 2) for i in range(100)] + [np.random.normal(5, 1, 2) for i in range(100)]
  model = MCores(k=20, beta=.5, epsilon=0.0)
  model.fit(a)
  
  result = model.predict(a)




Install
=======

This package uses distutils, which is the default way of installing
python modules.

To install for all users on Unix/Linux::

  sudo python setup.py build
  sudo python setup.py install



Dependencies
=======

python 2.7, scikit-learn


