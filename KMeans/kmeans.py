import numpy as np
from scipy.spatial.distance import cosine
import warnings

"""
K-Means clustering (Lloyd's algorithm)
Uses cosine similarity as a measure of similarity between two vectors and 
cosine distance (Cosine distance = 1 - Cosine similarity) to measure distances.
 
Takes following parameters:

    data : set of observations
    k    : number of clusters to be formed
    runs : number of times K-Means will run with different 
           centroids initialisation
    norm : whether samples have been normalised
    
Has the following attributes containing information about best kmeans run, 
meaning the one with lowest sum of intra-cluster distances:
    best_centroids   :  centroids of clusters
    best_clusters    :  clusters formed
    best_iterations  :  number of iterations
    best_intraDist
"""
class KMeans:
    
    """
    Initialises attributes containing input data but also the final clustering result.   
    """    
    def __init__(self,data,k,runs=10,norm=False):
        self.vectors = data  
        self.clusters_num = k
        self.runs = runs
        self.normalised = norm        
        self.best_centroids,self.best_clusters,self.best_iterations = None,None,None
        self.best_intraDist = np.infty
    
    """
    Check whether array of vectors is valid.    
    """    
    def _check_input(self):
        # Observations should be more than number k of clusters        
        if self.vectors.shape[0] < self.clusters_num:
            raise ValueError('Less samples (%d) than clusters (%d)'
                             % (self.vectors.shape[0],self.clusters_num))
    """
    Generates initial set of k cluster centers randomly.    
    """
    def _init_centroids(self):
        # Generate indices - arithmetic progression from 0 up to k-1
        indices = np.arange(self.vectors.shape[0]) 
        # Shuffle them: random list of indices        
        np.random.shuffle(indices)
        # Take first k indices and select the respective observations from vectors array
        self.centroids = self.vectors[indices[:self.clusters_num]]
    
    """
    Calculate new centers by finding mean values of clusters.
    """    
    def _recalculate_centroids(self):
        old_centroids = self.centroids.copy()
        centroids = np.empty(shape=(self.clusters_num,self.vectors.shape[1]))       
        for c in xrange(self.clusters_num):
            # Each time, samples belonging to certain cluster c are selected
            # Find mean of those and store it as centroid of c            
            np.mean(self.vectors[self.clusters == c],dtype=np.float64,axis = 0,out=centroids[c])        
        
        # In case there is centroid that consists only of zeros        
        if np.where(~centroids.any(axis=1))[0].size > 0:
            warnings.warn('Centroid contains only zeros: cosine distance undefined\n' \
            'Current run of k-means is terminated\n')
            return False
        
        # If vectors are normalised, normalise new centroids too
        if self.normalised:        
            self.centroids = centroids/(np.linalg.norm(centroids,axis=1)[:,np.newaxis])
        else:
            self.centroids = centroids
        return True
    
    """
    Compute clusters assigning samples to their nearest centroid.
    """    
    def _update_clusters(self):
        """ Cosine distance between normalised vectors """
        def cos_dist_normalised(c,v):
            return 1-np.sum(c*v,axis=0)
        
        distsum = 0
        ind = []
        # Select distance function based on whether vectors are normalised
        if self.normalised:
            distance = cos_dist_normalised
        else:
            distance = cosine
            
        # For every sample, find nearest center
        for v in self.vectors:
            # Find minimum distance and respective center index
            index, dist = min(enumerate([distance(c,v) for c in self.centroids]),key = lambda x: x[1])
            # Add to intra-cluster sum            
            distsum += dist
            # Result of clustering is a list that for every vector keeps the 
            # index of his centroid-cluster
            # Add index of cluster for current vector.             
            ind += [index]
         
        self.clusters = np.array(ind)
        old_intraDist = self.intraDist       
        self.intraDist = distsum
        # Return whether intraDist keeps monotonically decreasing and
        # loop should continue
        return (distsum < old_intraDist)

    """
    Single run of standard K-Means algorithm.
    """
    def _kmeans_single(self):
        self._check_input()
        # Compute initial set of k cluster centers
        self._init_centroids()    
        # Compute clusters. Loop continues while updated clusters have intraDist 
        # that monotonically decreases
        while self._update_clusters(): 
            # Calculate new centers of updated clusters
            # If it was unsuccessfull return false
            if not self._recalculate_centroids():
                return False
            self.iter_count+=1
        return True
        
    """
    Partitions data into clusters.
    Performs multiple runs of K-means and returns best result
    (minimum total distance from cluster centers or equivalently maximum similarity)
    """    
    def clustering(self):
        if self.runs < 1:
            raise ValueError('KMeans should run at least once')
        for _ in xrange(self.runs): 
            self.centroids, self.clusters = None, None
            self.intraDist = np.infty
            self.iter_count = 0;
            result = self._kmeans_single()
            # If kmeans was successfull and intraDist set a new record, 
            # update data of best run yet
            if result and (self.intraDist < self.best_intraDist):
                self.best_centroids = self.centroids.copy()
                self.best_clusters = self.clusters.copy()
                self.best_intraDist = self.intraDist
                self.best_iterations = self.iter_count
