import numpy as np
import pprint as pp

class CrowdTracker:
    def __init__(self, T, a_1, a_2, a_3, norm_a_1, norm_a_2, norm_a_3):
        self.previous_cluster_properties = None
        self.T = T
        self.a = {}
        self.a['center'] = a_1
        self.a['avg_directions'] = a_2
        self.a['area'] = a_3
        self.norm = {}
        self.norm['center'] = norm_a_1
        self.norm['avg_directions'] = norm_a_2
        self.norm['area'] = norm_a_3
        self.max_value = 10000
    #inverse mapping (this should ideally already be calculated and returned in DBSCAN for efficiency)
    def extract_cluster_properties(self, cluster_data, clustering):
        
        inverse_mapping = {c:[] for c in np.unique(clustering)}
        for i in range(len(cluster_data)):
            inverse_mapping[clustering[i]].append(cluster_data[i])

        if -1 in inverse_mapping.keys():
            inverse_mapping.pop(-1) 
        
        cluster_info = {}
        
        for c in inverse_mapping:
            mat = np.matrix(inverse_mapping[c])
            cluster_info[c] = {}
           # x = mat[:,[0,1]].mean(axis=0)[0,:]
            cluster_info[c]['center'] = list(map(int,mat[:,[0,1]].mean(axis=0).tolist()[0])) #consider using the other cords
            cluster_info[c]['avg_directions'] = np.asscalar(mat[:,5].mean(axis=0))
            cluster_info[c]['area'] = len(inverse_mapping[c])
            
        return cluster_info

    def S(self, cp_1, cp_2):
        d = {}
        d['center'] = np.sqrt(np.square(cp_1['center'][0] - cp_2['center'][0]) + np.square(cp_1['center'][1] - cp_2['center'][1]))
        phi = abs(cp_1['avg_directions'] - cp_2['avg_directions']) % 360
        d['avg_directions'] = 360 - phi if phi > 180 else phi
        d['area'] = abs(cp_1['area'] - cp_2['area'])
        result = 0

        for k in self.a.keys():
            #print(f'{k}: {(d[k]/self.norm[k])}')
            result += self.a[k] * (d[k]/self.norm[k])
        
      #  print(result)
        return result

    def evaluate_similarities(self, cps_1, cps_2):
        #TODO this might become sparse
        similarities = np.full((max(cps_1.keys())+1, max(cps_2.keys())+1), self.max_value, dtype='int16')
        
        for id_1 in cps_1.keys():
            for id_2 in cps_2.keys():
                similarities[id_1,id_2] = self.S(cps_1[id_1], cps_2[id_2])
                #if similarities[id_1,id_2] > self.T:
                  #  similarities[id_1,id_2] = self.max_value

        return similarities

    def create_mapping(self, similarities, clustering):
        mapping = {-1:-1}
        for _ in range(similarities.shape[0]):
            id_new, id_old = np.unravel_index(similarities.argmin(), similarities.shape)
            similarities[id_new, :] = self.max_value
            similarities[:, id_old] = self.max_value
           # print(f'id_new: {id_new}, id_old: {id_old}')
            mapping[id_new] = id_old
           # print(similarities)
        
        # Consider clusters, which were not mapped, and give them a new ID
        counter = 0
        for c in clustering:
            if c not in mapping.keys():
                mapping[c] = max(mapping.keys()) + 1 + counter
                counter += 1
        
        return mapping

    #TODO Deal with "noise" crowds as described in paper 
    def map_cluster_IDs(self, cluster_data, clustering):
        prev_cp = self.previous_cluster_properties
        cur_cp = self.extract_cluster_properties(cluster_data, clustering)

        if prev_cp is not None:
            similarities = self.evaluate_similarities(cur_cp, prev_cp)
            # print(similarities)
            mapping = self.create_mapping(similarities, cur_cp.keys())
            print(mapping)
            new_clustering = np.array([mapping[i] for i in clustering])
            return new_clustering, cur_cp
        else:
            self.previous_cluster_properties = cur_cp
            return clustering, cur_cp

    # #TODO Deal with "noise" crowds as described in paper 
    # def map_cluster_IDs_alt(self, cluster_data, clustering):
    #     prev_cp = self.previous_cluster_properties
    #     cur_cp = self.extract_cluster_properties(cluster_data, clustering)
    #     if prev_cp is not None:
    #         # Do cross checking to unify cluster IDs, map cluster IDs from current
    #         # frame to cluster IDs from old frame OR to a new cluster ID
    #         mapping = {-1: -1}
    #         already_mapped = set() 
    #         for cur_id in cur_cp.keys():
    #             # Assume that initially the current crowd is new => it gets a new (garantueed unique) ID
    #             print((already_mapped))
    #             candidate_id, candidate_similarity = max(list(prev_cp.keys())+list(already_mapped)+[0])+1 , 0
    #             for prev_id in prev_cp.keys() - already_mapped:
    #                 similarity = self.S(prev_cp[prev_id], cur_cp[cur_id])
    #                 if similarity < self.T and similarity < candidate_similarity:
    #                     # New candidate found
    #                     candidate_id, candidate_similarity = prev_id, similarity
    #                     print(already_mapped)
    #             # Map cur cluster ID to old OR new cluster ID
    #             mapping[cur_id] = candidate_id
    #             already_mapped.add(candidate_id)
                
    #         pp.pprint(mapping)
    #         new_clustering = np.array([mapping[i] for i in clustering])
    #         return new_clustering, cur_cp
    #     else:
    #         self.previous_cluster_properties = cur_cp
    #         return clustering, cur_cp
