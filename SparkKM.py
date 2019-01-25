from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
import numpy as np 
from scipy.spatial import distance
from random import sample
class KMeans:
    def __init__(self, X, init='random'):
        if init=='random':
            
        self.centroides = sample(list(X),10)
        self.X=X


    def fit(self):
        spark = SparkSession.builder.appName("spark_paral").getOrCreate()
        sparkContext= spark.sparkContext
        data=sparkContext.parallelize(self.X)
        distancias_minimas = data.map(self.diferencia_minima)
        #print(self.X,"sdsd")
        #print(f,"collect")
        return f

    def diferencia_minima(self,x):    
        
        for centroide in range(len(self.centroides)):            
            if centroide==0:
                dist_minima=Vectors.squared_distance(np.asarray(x),np.asarray(self.centroides[centroide]))                
            else:
                if dist_minima > x.squared_distance(self.centroides(centroide)):
                    dist_minima=x.squared_distance(self.centroides(centroide))
        return 
    

def main():
    

    X=np.random.rand(10,2)*100
    kmm=KMeans(X)
    print(kmm.fit())
    print("hecho")


if __name__ == "__main__":
    main()