import numpy as np

class CrowdTracker():
    
    def __init__(self, T, a_1, a_2):
        self.previous_cluster_properties = None
        self.T = T
        self.a_1 = a_1
        self.a_2 = a_2

    #inverse mapping (this should ideally already be calculated and returned in DBSCAN for efficiency)
    def label_specific_properties(cluster_data, clustering):
        
        inverse_mapping = {c:[] for c in np.unique(clustering)}
        for i in range(len(cluster_data)):
            inverse_mapping[clustering[i]].append(cluster_data[i])
            
        cluster_info = {}
        
        for c in inverse_mapping:
            mat = np.matrix(inverse_mapping[c])
            cluster_info[c] = {}
            cluster_info[c]['center'] = mat[:,[0,1]].mean(axis=0).tolist() #consider using the other cords
            cluster_info[c]['avg_directions'] = np.asscalar(mat[:,5].mean(axis=0))
        
        return cluster_info

    #TODO finish this
    def S(ci_1, c_i2):
        return self.a1 * (ci1) + self.a2

    def return_adjusted_coloring(self, cluster_data, clustering):
        cur_clustering_properties = self.label_specific_properties(cluster_data, clustering)
        if self.previous_cluster_properties is not None:
            # Do cross checking to unify cluster IDs
            for c_prev in self.previous_cluster_properties:
                for c_new in cur_clustering_properties:
                    x = self.S(self.previous_cluster_propertie[c_prev], cur_clustering_properties[c_new])
        else:
            return clustering

