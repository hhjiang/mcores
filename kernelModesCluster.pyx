import numpy as np
cimport numpy as np
from sklearn.neighbors import KDTree, BallTree
import scipy
import math
import sys


cdef extern from "mutual_knn.h":
	void compute_mutual_knn(int n, int k, int d,
                    double * radii,
                    int * neighbors,
                    double beta,
                    double epsilon,
                    int * result)

cdef compute_mutual_knn_np(n, k, d,
					np.ndarray[double,  ndim=1, mode="c"] radii,
					np.ndarray[np.int32_t, ndim=2, mode="c"] neighbors,
					beta,
					epsilon,
					np.ndarray[np.int32_t, ndim=1, mode="c"] result):

	compute_mutual_knn(n, k, d,
    					<double *> np.PyArray_DATA(radii),
    					<int *> np.PyArray_DATA(neighbors),
    					beta, epsilon,
    					<int *> np.PyArray_DATA(result))


cdef extern from "hausdorff_computation.h":
	void get_closest_clusters(int n, int d, int n_clusters,
					int n_clustered_points,
					double * clustered_points,
					int * memberships,
					double * points,
					int * result,
					double * minimax_distances)

	void precompute_pairwise_distances(int n, int d,
									double * points,
									double * result)

	void get_closest_clusters_precomputed(int n,
									int n_clusters,
									int * memberships,
									double * distances,
									int * result)


cdef extern get_closest_clusters_np(n, d, n_clusters,
					n_clustered_points,
					np.ndarray[double, ndim=2, mode="c"] clustered_points,
					np.ndarray[np.int32_t, ndim=1, mode="c"] memberships,
					np.ndarray[double, ndim=2, mode="c"] points,
					np.ndarray[np.int32_t, ndim=1, mode="c"] result,
					np.ndarray[double, ndim=1, mode="c"] minimax_distances):

	get_closest_clusters(n, d, n_clusters,
						n_clustered_points, 
						<double *> np.PyArray_DATA(clustered_points),
						<int *> np.PyArray_DATA(memberships),
						<double *> np.PyArray_DATA(points),
						<int *> np.PyArray_DATA(result),
						<double *> np.PyArray_DATA(minimax_distances))



cdef extern precompute_pairwise_distances_np(n, d,
					np.ndarray[double, ndim=2, mode="c"] points,
					np.ndarray[double, ndim=1, mode="c"] result):

	precompute_pairwise_distances(n, d,
			<double *> np.PyArray_DATA(points),
			<double *> np.PyArray_DATA(result))



cdef extern get_closest_clusters_precomputed_np(n, n_clusters,
		np.ndarray[np.int32_t, ndim=1, mode="c"] memberships,
		np.ndarray[double, ndim=1, mode="c"] distances,
		np.ndarray[np.int32_t, ndim=1, mode="c"] result):

	get_closest_clusters_precomputed(n, n_clusters,
		<int *> np.PyArray_DATA(memberships),
		<double *> np.PyArray_DATA(distances),
		<int *> np.PyArray_DATA(result))







class MCores:
	"""Perform Kernel Modes Clustering of data

	Parameters
	----------
	
	k: The number of neighbors (i.e. the k in k-NN density)

	beta: Ranges from 0 to 1. We choose points that have kernel density of at
		least (1 - beta) * F where F is the mode of the empirical density of
		the cluster

	epsilon: For pruning. Sets how much deeper in the cluster tree to look
		in order to connect clusters together. Must be at least 0.

	cluster_threshold: Determines the minimum threshold distance for us
		to classify points to its corresponding closest (Hausdorff distance)
		estimated mode. All other points do not get assigned to a cluster.
		If this paramter is None, then all points get assigned to some
		cluster. 


	Attributes
	----------

	n_clusters: number of clusters fitted

	cluster_map: a map from the cluster (zero-based indexed) to the list of points
		in that cluster

	"""



	def __init__(self, k, beta,
					epsilon=0,
					cluster_threshold=None,
					kernel="knn",
					ann="kdtree"):
		self.k = k
		self.beta = beta
		self.epsilon = epsilon
		self.cluster_threshold = cluster_threshold
		self.kernel = kernel
		self.n_clusters = 0
		self.cluster_map = {}
		self.clustered_points = None
		self.memberships = None
		self.ann = ann


	def precompute(self, X):
		"""
		computes knn density estimates and distances
		"""

		X = np.array(X)
		n, d = X.shape
		knn_density = None
		neighbors = None

		if self.ann == "kdtree":
			kdt = KDTree(X, metric='euclidean')
			query_res = kdt.query(X, k=self.k)

		elif self.ann == "balltree":
			balltree = BallTree(X, metric='euclidean')
			query_res = balltree.query(X, k=self.k)

		self.neighbors_data = query_res[1]
		self.distances_data = query_res[0]

		self.pairwise_distances = np.zeros(n * n, dtype=np.float64)

		precompute_pairwise_distances_np(n, d,
					X,
					self.pairwise_distances)


		return

	def fit_predict(self, X, precompute=True):
		"""
		build neighbor graphs and find the clusters
		"""
		X = np.array(X)
		n, d = X.shape

		knn_radius = self.distances_data[:, self.k - 1]
		neighbors = self.neighbors_data

		memberships = np.zeros(n, dtype=np.int32)
		neighbors = np.ndarray.astype(neighbors, dtype=np.int32)
		knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)


		compute_mutual_knn_np(n, self.k, d,
    						knn_radius,
    						neighbors,
    						self.beta, self.epsilon,
    						memberships)

		self.n_clusters = np.unique(memberships[np.where(memberships >= 0)]).shape[0]



		for i in xrange(self.n_clusters):
			self.cluster_map[i] = []

		for i in xrange(len(memberships)):
			if memberships[i] >= 0:
				self.cluster_map[memberships[i]].append(X[i,:])

		self.memberships = memberships[np.where(memberships >= 0)]
		self.raw_memberships = memberships

		self.raw_memberships = np.ndarray.astype(self.raw_memberships, dtype=np.int32)

		self.clustered_points = X[np.where(memberships >= 0), :][0]


		if precompute:

			result = np.zeros(n, dtype=np.int32)
			minimax_distances = np.zeros(n, dtype=np.float64)

			get_closest_clusters_precomputed_np(n, self.n_clusters,
				self.raw_memberships,
				self.pairwise_distances,
				result) 

			return result

		else:
			return self.predict(X)


	def fit(self, X):
		"""
		Determines the clusters in two steps.
		First step is to compute the knn density estimate and
		distances. This is done using kd tree
		Second step is to build the knn neighbor graphs
		Updates the cluster count and membership attributes

		Parameters
		----------
		X: Data matrix. Each row should represent a datapoint in 
			euclidean space
		"""
		X = np.array(X)
		n, d = X.shape
		knn_density = None
		neighbors = None

		if self.ann == "kdtree":
			kdt = KDTree(X, metric='euclidean')
			query_res = kdt.query(X, k=self.k)
			knn_radius = query_res[0][:, self.k-1]
			neighbors = query_res[1]

		elif self.ann == "balltree":
			balltree = BallTree(X, metric='euclidean')
			query_res = balltree.query(X, k=self.k)
			knn_radius = query_res[0][:, self.k - 1]
			neighbors = query_res[1]

		memberships = np.zeros(n, dtype=np.int32)
		neighbors = np.ndarray.astype(neighbors, dtype=np.int32)
		knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)

		compute_mutual_knn_np(n, self.k, d,
    						knn_radius,
    						neighbors,
    						self.beta, self.epsilon,
    						memberships)
		
		self.n_clusters = np.unique(memberships[np.where(memberships >= 0)]).shape[0]

		for i in xrange(self.n_clusters):
			self.cluster_map[i] = []

		for i in xrange(len(memberships)):
			if memberships[i] >= 0:
				self.cluster_map[memberships[i]].append(X[i,:])

		self.memberships = memberships[np.where(memberships >= 0)]
		self.clustered_points = X[np.where(memberships >= 0), :][0]
		self.raw_memberships = memberships


	def predict(self, X):
		"""
		Takes in a matrix of points and returns the cluster
		memberships, zero indexed. If no cluster membership, then that 
		point will have result -1
		"""

		X = np.array(X, dtype=np.float64)
		n, d = X.shape
		result = np.zeros(n, dtype=np.int32)
		minimax_distances = np.zeros(n, dtype=np.float64)


		get_closest_clusters_np(n, d,
							self.n_clusters,
							self.clustered_points.shape[0],
							self.clustered_points,
							self.memberships,
							X,
							result,
							minimax_distances
						)

		return result







