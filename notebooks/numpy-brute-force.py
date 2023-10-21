import numpy as np
import h5py
import timeit

with h5py.File("core/.fashion-mnist-784-euclidean.hdf5") as hfp:
# with h5py.File(".glove-100-angular.hdf5") as hfp:
    dataset = hfp['train'][:]
    queries = hfp['test'][:]
    dataset_sqnorms = np.linalg.norm(dataset, axis=1)**2
    queries_sqnorms = np.linalg.norm(queries, axis=1)**2


def brute_force(dataset, query, dataset_sqnorms, query_sqnorm, k):
    # compute the squared euclidean distances using the formula
    #   ||x||^2 + ||y||^2 - 2xy
    dists = dataset_sqnorms + query_sqnorm - 2*np.dot(dataset, query)
    # find the smallest k
    dists = np.partition(dists, k)
    # return them in sorted order
    return np.sort(dists[:k])


qidx = 0
k = 100

print(brute_force(dataset, queries[qidx,:], dataset_sqnorms, queries_sqnorms[qidx], k))

runs = 1000
elapsed = timeit.timeit(lambda: brute_force(dataset, queries[qidx,:], dataset_sqnorms, queries_sqnorms[qidx], k), number=runs)
print("average query time", (elapsed / runs) * 1000, "ms")

