import numpy as np

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
def S(ci1, ci2):
    a1 = 0.8
    a2 = 0.2
    
    
    
    return a1 * (ci1)