from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
import numpy as np 
from scipy.spatial import distance
from random import sample
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, X, k, init='random'):
        self.X=X
        self.k=k
        if init=='random':            
            self.centroides = np.asarray(sample(list(self.X),self.k))
        

    def fit(self):
        spark = SparkSession.builder.appName("spark_paral").getOrCreate()
        sparkContext= spark.sparkContext
        Puntos=sparkContext.parallelize(self.X)        
        self.llave_centroides = Puntos.map(self.diferencia_minima)
        nuevos_centroides=Puntos.map(self.nuevos_centroides)
        #print(self.X,"sdsd")
        #print(np.asarray(Puntos.collect()),"collect")
        #print(llave_centroides.collect(),"collect")
        return llave_centroides

    def nuevos_centroides(self,x):
        for centroide in 

    def getCentroides(self):
        return self.centroides

    def diferencia_minima(self,x):            
        for centroide in range(self.k):            
            if centroide==0:
                dist_minima=Vectors.squared_distance(x,self.centroides[centroide])
                llave_centroide=centroide
            else:
                if dist_minima > Vectors.squared_distance(x,self.centroides[centroide]):
                    llave_centroide=centroide
        return llave_centroide
        
def main():
    

    X=np.random.rand(10,2)*100
    kmm=KMeans(X,3)
    print(kmm.fit().collect())
    print("hecho")
    print(kmm.getCentroides())


    colorlist=[ '#051b30', '#8bb6de', '#434f08', '#4f1008', '#c8493a', '#e7209b','#050100', '#1e7614', '#76a071', '#745a6a']

    #print(km.getCentroides())
    centroides=np.array(kmm.getCentroides())
    
    for x in range(X.shape[0]):        
        plt.plot(X[x][0],X[x][1], marker='o', color='b', ls='')
    #plt.axis([0, 5, 0, 5])
    for x in range(centroides.shape[0]):        
        plt.plot(centroides[x][0],centroides[x][1], marker='o', color='r', ls='')
    plt.show()
    


if __name__ == "__main__":
    main()
    