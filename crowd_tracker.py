import numpy as np
import pprint as pp

class CrowdTracker:
    def __init__(self, T, a_1, a_2):
        self.previous_cluster_properties = None
        self.T = T
        self.a_1 = a_1
        self.a_2 = a_2
    #inverse mapping (this should ideally already be calculated and returned in DBSCAN for efficiency)
    def label_specific_properties(self, cluster_data, clustering):
        
        inverse_mapping = {c:[] for c in np.unique(clustering)}
        for i in range(len(cluster_data)):
            inverse_mapping[clustering[i]].append(cluster_data[i])

        if -1 in inverse_mapping.keys():
            inverse_mapping.pop(-1) 
        
        cluster_info = {}
        
        for c in inverse_mapping:
            mat = np.matrix(inverse_mapping[c])
            cluster_info[c] = {}
            cluster_info[c]['center'] = mat[:,[0,1]].mean(axis=0).tolist() #consider using the other cords
            cluster_info[c]['avg_directions'] = np.asscalar(mat[:,5].mean(axis=0))
        
        return cluster_info

    def S(self, ci_1, ci_2):
        distance = np.sqrt(np.square(ci_1['center'][0][0] - ci_1['center'][0][0]) + np.square(ci_1['center'][0][1] - ci_1['center'][0][1]))
        return self.a_1 * distance + self.a_2 * abs(ci_1['avg_directions'] - ci_2['avg_directions'])

    #TODO Deal with "noise" crowds as described in paper 
    def map_cluster_IDs(self, cluster_data, clustering):
        prev_cp = self.previous_cluster_properties
        cur_cp = self.label_specific_properties(cluster_data, clustering)
        if prev_cp is not None:
            # Do cross checking to unify cluster IDs, map cluster IDs from current
            # frame to cluster IDs from old frame OR to a new cluster ID
            mapping = {-1: -1}
            for cur_id in cur_cp.keys():
                if cur_id == -1: continue;
                # Assume that initially the current crowd is new => it gets a new (garantueed unique) ID
                candidate_id, candidate_similarity = max(list(prev_cp.keys())+[0])+1 , 0
                for prev_id in prev_cp.keys():
                    similarity = self.S(prev_cp[prev_id], cur_cp[cur_id])
                    if similarity > self.T and similarity > candidate_similarity:
                        # New candidate found
                        candidate_id, candidate_similarity = prev_id, similarity
                # Map cur cluster ID to old OR new cluster ID
                mapping[cur_id] = candidate_id
            
            pp.pprint(mapping)
            new_clustering = np.array([mapping[i] for i in clustering])
            return new_clustering, cur_cp
        else:
            self.previous_cluster_properties = cur_cp
            return clustering, cur_cp

