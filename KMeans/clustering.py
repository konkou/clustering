import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from kmeans import KMeans
import time
import optparse

""" Argumens parsing """
parser = optparse.OptionParser()

parser.add_option('--d', action="store", dest="dataset",
                  help="Dataset file name.")
parser.add_option("--k",
              dest="k", type="int",
              help="Number of clusters to be formed.")
parser.add_option("--r",
              dest="runs", type="int",
              help="Number of times K-Means will run.")
parser.add_option('--n', action="store_true", default=False,
              dest = "norm",
              help="Normalise data before K-Means clustering")
parser.add_option("--c",
              dest="n_comp", type="int",
              help="Number of components for LSI")

parser.print_help()
              
op, remainder = parser.parse_args()


start = time.time()
# Preprocessing
print "Preprocessing begins..."
vectorizer=TfidfVectorizer(stop_words='english')
df=pd.read_csv(op.dataset,sep="\t")
X = vectorizer.fit_transform(df['Content'])

# Find rows with non zero values and select them
num_nonzeros = np.diff(X.indptr)
X = X[num_nonzeros != 0]
  

svd=TruncatedSVD(n_components=op.n_comp, random_state=42)
# Option to normalise
if op.norm:
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
else:
    X = svd.fit_transform(X)

end = time.time()
print "Preprocessing ends in %f seconds" %(end - start)
km = KMeans(X,op.k,op.runs,op.norm)

print "KMeans begins..."
start = time.time()
km.clustering()
end = time.time()
print "Kmeans completed in %f seconds" %(end - start)

centroids = km.best_centroids
clusters = km.best_clusters
intraSum = km.best_intraDist

print "KMeans ended with best sum of distances: "+str(intraSum)
print "Computing statistics... "

clusters_info = [None] * op.k
for i in xrange(op.k):
    categ = df['Category'][num_nonzeros != 0][clusters == i]
    # Get relative frequencies of categories in cluster i
    stats = categ.value_counts(normalize=True)
    clusters_info[i]=dict(stats)



cluster_tags = ["Cluster"+str(i+1) for i in xrange(op.k)]
for i,c in enumerate(clusters_info):
    c['']=cluster_tags[i]

# Create dataframe and csv
rdf = pd.DataFrame(clusters_info) 
rdf = rdf[['']+df.Category.unique().tolist()]
rdf.to_csv('clustering_KMeans.csv',mode = 'w',na_rep='0',float_format='%.3f', index=False)
print "Results exported in file clustering_KMeans.csv"
