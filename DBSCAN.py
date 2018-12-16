import numpy as np
import math
from sklearn.cluster import DBSCAN as dbs

maxAngleDif = 25
maxMagnDif  = 10

def regionQuery(setOfPoints, point, eps):
    def euclidianDistOfinitPoints(firstPoint, secondPoint):
            x =(firstPoint[0] - secondPoint[0])*(firstPoint[0] - secondPoint[0]) + (firstPoint[1] - secondPoint[1])*(firstPoint[1] - secondPoint[1])
            x = math.sqrt(x)
            return x
    epsNeighbourhood = []
    for i in range (len(setOfPoints)):
        dist = euclidianDistOfinitPoints(point, setOfPoints[i])
        difMagn = point[6]-setOfPoints[i][6]
        difAngle = point[7]-setOfPoints[i][7]
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
    labels = []
    for i in range(len(setOfPoints)):
        labels.append(-2)
    for i in range(len(setOfPoints)):
        point = setOfPoints[i]
        if labels[i] == -2:
            if ExpandCluster(setOfPoints,labels, i, point,Id,Eps,MinPts):
                Id = Id + 1
    return np.array(labels)

