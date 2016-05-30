# K-Means

Standard K-Means clustering algorithm implemented in Python for Data Mining course.  
Preprocessing and analysis of the result is handled in `clustering.py`  
Results are exported into a .csv file

## How to run 

```sh
$ python clustering.py --d Dataset_file --k Number_of_clusters --r Number_of_runs --c Number_of_LSI_components
```
Use `--n` to normalise data before clustering

List of command line arguments:

```sh
$ python clustering.py --h
```
To perform clustering without any of the provided preprocessing use KMeans class and its clustering method.
For example:

```python
km = KMeans(X,op.k,op.runs,op.norm)
km.clustering()
``` 

#### TODO
Implement 
* K-Means++
* Hierarchical clustering  

#### Environment
Tested in Ubuntu 14.04, Python 2.7.6 
