#include <map>
#include <stack>
#include <set>
#include <vector>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <fstream>
using namespace std;

struct Node {
    /*
    Node struct for our k-NN Graph
    */
    int index;
    int rank;
    Node * parent;
    set <Node * > children;
    Node(int idx) {
    	index = idx;
    	rank = 0;
    	parent = NULL;
    	children.clear();
    }

};

struct Graph {
    /*
    k-NN graph struct.
    Allows us to build the graph one node at a time
    */
    vector <Node *> nodes;
    map <int, Node * > M;
    set <Node * > intersecting_sets;
    Graph() {
        M.clear();
        intersecting_sets.clear();
        nodes.clear();
    }

    Node * get_root(Node * node) {

         if (node->parent != NULL) {
         	node->parent->children.erase(node);
         	node->parent = get_root(node->parent);
         	node->parent->children.insert(node);
         	return node->parent;
         } else {
            return node;
         }    	
    }

    void add_node(int idx) {
       nodes.push_back(new Node(idx));
       M[idx] = nodes[nodes.size() - 1];
    }

    void add_edge(int n1, int n2) {
    	Node * r1 = get_root(M[n1]);
    	Node * r2 = get_root(M[n2]);
    	if (r1 != r2) {
    		if (r1->rank > r2->rank) {
    			r2->parent = r1;
    			r1->children.insert(r2);
    			if (intersecting_sets.count(r2)) {
    				intersecting_sets.erase(r2);
    				intersecting_sets.insert(r1);
    			}
    		} else {
    			r1->parent = r2;
    			r2->children.insert(r1);
    			if (intersecting_sets.count(r1)) {
    				intersecting_sets.erase(r1);
    				intersecting_sets.insert(r2);
    			}

    			if (r1->rank == r2->rank) {
    				r2->rank++;
    			}
    		}
    	}
    }

    vector <int> get_connected_component(int n) {
        Node * r = get_root(M[n]);
        vector <int> L;
        stack <Node * > s;
        s.push(r);
        while (!s.empty()) {
            Node * top = s.top(); s.pop();
            L.push_back(top->index);
            for (set<Node * >::iterator it = top->children.begin();
            	                    it != top->children.end();
            	                    ++it) {
                s.push(*it);
    		}
    	}
    	return L;
    }


    bool component_seen(int n) {
        Node * r = get_root(M[n]);
        if (intersecting_sets.count(r)) {
             return true;
        }
        intersecting_sets.insert(r);
        return false;
    }

    int GET_ROOT(int idx) {
    	Node * r = get_root(M[idx]);
    	return r->index;
    }

    vector <int> GET_CHILDREN(int idx) {
    	Node * r = M[idx];
    	vector <int> to_ret;
    	for (set<Node *>::iterator it = r->children.begin();
    		                       it != r->children.end();
    		                       ++it) {
    		to_ret.push_back((*it)->index);
    	}
    	return to_ret;
    }

};




void compute_mutual_knn(int n, int k,
                    double * densities,
                    int * neighbors,
                    double beta,
                    double epsilon,
                    int * result) {
    /* Given the kNN density and neighbors
        We build the k-NN graph / cluster tree and return the estimated modes.
        Note that here, we don't require the dimension of the dataset 
        Returns array of estimated mode membership, where each index cosrresponds
        the respective index in the density array. Points without
        membership are assigned -1 */

    vector<pair <double, int> > knn_densities(n);
    vector <set <int> > knn_neighbors(n);

    /*freopen("debug_notes", "w", stdout);
    printf("Neighbors");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; ++j) {
            printf("%d ", neighbors[i * k + j]);
        }
        printf("\n");

    }

    printf("densities");
    for (int i = 0; i < n; i++) {
        printf("%f\n", densities[i]);
    }
    fclose(stdout);*/


    for (int i = 0; i < n; ++i) {
        knn_densities[i].first = densities[i];
        knn_densities[i].second = i;

        for (int j = 0; j < k; ++j) {
            knn_neighbors[i].insert(neighbors[i * k + j]);
        }
    }

    int m_hat[n];
    int cluster_membership[n];
    int n_chosen_points = 0;
    int n_chosen_clusters = 0;
    sort(knn_densities.begin(), knn_densities.end());
    reverse(knn_densities.begin(), knn_densities.end());

    Graph G = Graph();

    int last_considered = 0;
    int last_pruned = 0;
    
    for (int i = 0; i < n; ++i) {
        while (last_pruned < n && knn_densities[last_pruned].first > knn_densities[i].first / (1. + epsilon)) {

            G.add_node(knn_densities[last_pruned].second);

            for (set <int>::iterator it = knn_neighbors[knn_densities[last_pruned].second].begin();
                                    it != knn_neighbors[knn_densities[last_pruned].second].end();
                                 ++it) {
                if (G.M.count(*it)) {
                  if (knn_neighbors[*it].count(knn_densities[last_pruned].second)) {
                        G.add_edge(knn_densities[last_pruned].second, *it);
                    }
             
                }

            }
            last_pruned++;
        }


        
        while (knn_densities[last_considered].first - knn_densities[i].first 
               > beta * knn_densities[last_considered].first) {

            if (!G.component_seen(knn_densities[last_considered].second)) {
                vector <int> res = G.get_connected_component(knn_densities[last_considered].second);
                for (int j = 0; j < res.size(); j++) {
                    if (densities[res[j]] >= knn_densities[i].first) {
                        cluster_membership[n_chosen_points] = n_chosen_clusters;
                        m_hat[n_chosen_points++] = res[j];
                    }

                }
                n_chosen_clusters++;
            }
            last_considered++;
        }
    }

    for (int i = 0; i < n; ++i) {
        result[i] = -1;
    }
    
    for (int i = 0; i < n_chosen_points; ++i) {
        result[m_hat[i]] = cluster_membership[i];
    }

}



