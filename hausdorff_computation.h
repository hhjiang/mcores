#include <vector>
#define MAX_DOUBLE 1000000000

using namespace std;




double get_euclidean(double * a, double * b, int d) {
	double dist = 0.0;
	for (int i = 0; i < d; ++i) {
		dist += (*(a + i) -  *(b + i)) * (*(a + i) - *(b + i));
	}
	return dist;
}


/* To do: create a version that is fast */
double get_hausdorff_distance(int point_idx,
							int d, 
							vector <int> &clustered_point_idxs,
							double * points,
							double * clustered_points) {

	double min_distance = MAX_DOUBLE;
	double dist;
	for (int i = 0; i < clustered_point_idxs.size(); ++i) {

		dist = get_euclidean(points + point_idx * d,
							clustered_points + clustered_point_idxs[i] * d,
							d);

		min_distance = min_distance > dist ? dist : min_distance; 
	}

	return min_distance;


}



void precompute_pairwise_distances(int n, int d,
								double * points,
								double * result) {
	for (int i = 0; i < n; ++i) {
		for (int j = i; j < n; ++j) {
			double dist = get_euclidean(points + i * d, points + j * d, d);
			result[i * n + j] = dist;
			result[j * n + i] = dist;
		}
	}
}



void get_closest_clusters_precomputed(int n,
									int n_clusters,
									int * memberships,
									double * distances,
									int * result) {

	vector <int> * M = new vector <int>[n_clusters];
	for (int i = 0; i < n; ++i) {
		result[i] = -1;
		if(memberships[i] >= 0) {
			M[memberships[i]].push_back(i);
			result[i] = memberships[i];
		}


	}

	for (int i = 0; i < n; ++i) {
		if (result[i] < 0) {
			double minimax = MAX_DOUBLE;
			for (int cluster = 0; cluster < n_clusters; ++cluster) {
				for (int p = 0; p < M[cluster].size(); ++p) {
					if (minimax > distances[M[cluster][p] * n + i]) {
						minimax = distances[M[cluster][p] * n + i];
						result[i] = cluster;
					}
				}

			}
		}
	}

}


void get_closest_clusters(int n, int d, int n_clusters,
					int n_clustered_points, 
					double * clustered_points,
					int * memberships,
					double * points,
					int * result,
					double * minimax_distances) {

	vector <int> * M = new vector <int>[n_clusters];
	double res;


	for (int i = 0; i < n_clustered_points; ++i) {
		M[memberships[i]].push_back(i);
	}


	for (int point_idx = 0; point_idx < n; ++point_idx) {
		result[point_idx] = -1;
		minimax_distances[point_idx] = MAX_DOUBLE;
		
		for (int cluster = 0; cluster < n_clusters; ++cluster) {
			res = get_hausdorff_distance(point_idx, d, M[cluster], points, clustered_points);
			if (minimax_distances[point_idx] > res) {
				minimax_distances[point_idx] = res;
				result[point_idx] = cluster;
			}
		}

	}


}