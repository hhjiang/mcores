
stly taken from http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
 

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import time

from MCores import *


from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
random.seed(1234)
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None


halving_matrix = np.zeros((n_samples, 2))
for i in xrange(n_samples):
    r = random.random()
    if r < 0.5:
        halving_matrix[i, 0] = np.random.normal(0, 1) - 2
        halving_matrix[i, 1] = np.random.normal(0, 1) - 2
    else:
        halving_matrix[i, 0] = np.random.normal(0, 1) + 1
        halving_matrix[i, 1] = np.random.normal(0, 1) + 1

halving = halving_matrix, None

three_modes_matrix = np.zeros((n_samples, 2))

for i in xrange(n_samples):
    r = random.random()
    if r < 0.6:
        three_modes_matrix[i, 0] = np.random.uniform(-1, 1)
        three_modes_matrix[i, 1] = np.random.uniform(0, 1)
    elif r < 0.8:
        three_modes_matrix[i, 0] = np.random.normal(0, 0.1) + 1.5
        three_modes_matrix[i, 1] = np.random.normal(0, 0.1) - 0.5
    else:
        three_modes_matrix[i, 0] = np.random.normal(0, 0.1) + 1.5
        three_modes_matrix[i, 1] = np.random.normal(0, 0.1) + 0.5

three_modes = three_modes_matrix, None


five_modes_matrix = np.zeros((n_samples, 2))
five_modes_centers = [(0, 0), (2, 2), (-2, 2)]
five_modes_csum =[0.33, 0.66, 1.01]
for i in xrange(n_samples):
    t = random.random()
    if t < 0.3:
        five_modes_matrix[i, 0] = np.random.uniform(-2, 2)
        five_modes_matrix[i, 1] = np.random.normal(0, 0.42) - 2
        continue

    r = random.random()
    for j, prob in enumerate(five_modes_csum):
        if r < prob:
            five_modes_matrix[i, 0] = np.random.normal(0, 0.5) + five_modes_centers[j][0]
            five_modes_matrix[i, 1] = np.random.normal(0, 0.5) + five_modes_centers[j][1]
            break


five_modes = five_modes_matrix, None



colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

clustering_names = [
    'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'DBSCAN', 'Birch', 'MCores']

plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

datasets = [noisy_circles, noisy_moons, blobs, no_structure, halving, five_modes, three_modes]
n_clusters_data = [2, 2, 3, 1, 2, 4, 3]


for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    n_clusters = n_clusters_data[i_dataset]
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
    ward = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                           connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=n_clusters,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=.2)
    
    k = 30
    knn_modes = MCores(k=k, beta=3.5 / math.sqrt(k), epsilon=0.)
    
    
    affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)

    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=n_clusters,
        connectivity=connectivity)

    birch = cluster.Birch(n_clusters=n_clusters)
    clustering_algorithms = [
        two_means, affinity_propagation, ms, spectral, ward, average_linkage,
        dbscan, birch, knn_modes]

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # plot
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
