import numpy as np

maxAngleDif = 25
maxMagnDif  = 10

def regionQuery(setOfPoints, point, eps):
    def euclidianDistOfinitPoints(x1, x2):
        x = np.square(x1[0] - x2[0]) + np.square(x1[1] - x2[1])
        x = np.sqrt(x)
        return x
    epsNeighbourhood = []
    for i in range (len(setOfPoints)):
        dist = euclidianDistOfinitPoints(point[0:2], setOfPoints[i])
        difMagn = point[4]-setOfPoints[i][4]
        difAngle = point[5]-setOfPoints[i][5]
        isTrue =dist <= eps and abs(difMagn)<=maxMagnDif and abs(difAngle)<=maxAngleDif
        if isTrue:
            epsNeighbourhood.append(setOfPoints[i])
    return epsNeighbourhood

#label -1 -> NOISE
#label -2 -> Unclassified
#label 1.. -> Classified
def ExpandCluster(setOfPoints, labels, pointId, point, ClId, Eps, MinPts):
    seeds = regionQuery(setOfPoints, point, Eps)
    if len(seeds)<MinPts:
        labels[pointId] = -1
        return False
    else:
        labels[pointId] = ClId
        seeds.remove(point)
        while len(seeds) != 0:
            currentP = seeds.pop()
            
            result = regionQuery(setOfPoints,currentP,Eps)
            if len(result) >= MinPts:
                for i in range(len(result)):
                    resultP = result[i]
                    indexP = setOfPoints.index(resultP)
                    if labels[indexP] in [-2,-1]:
                        if labels[indexP] == -2:
                            seeds.append(resultP)
                        labels[indexP] = ClId
        return True

def DBSCAN(setOfPoints,Eps,MinPts):
    Id = 0
    labels = [-2] * len(setOfPoints)
    for i in range(len(setOfPoints)):
        point = setOfPoints[i]
        if labels[i] == -2:
            if ExpandCluster(setOfPoints,labels, i, point,Id,Eps,MinPts):
                Id = Id + 1
    return np.array(labels)











