"""
This module implements the K-means clustering algorithm using a given distance metric.
Available distance metrics include 

- euclidean
- cosine
- manhattan
- chebyshev

Functions:
- distance(x1, x2, metric="euclidean"): Computes the distance between two points using the specified metric.
- kmeans(data, k, metric="euclidean", max_iters=100): Performs K-means clustering on the given data.

Example usage:
    from kmeans.kmeans import kmeans
    data = [[1, 2], [1, 1], [2, 3], [8, 7], [9, 8], [7, 9]]
    k = 2
    centroids, clusters = kmeans(data, k, metric="euclidean")



- Sample data: data = np.array([[1, 2], [1, 1], [2, 3], [8, 7], [9, 8], [7, 9]])
- Number of clusters: k = 2
- Run K-means with Euclidean distance metric: centroids_euclidean, clusters_euclidean = kmeans(data, k, metric="euclidean")
- Run K-means with Cosine distance metric: centroids_cosine, clusters_cosine = kmeans(data, k, metric="cosine")
"""


import random
import math
from collections import defaultdict


def distance(x1, x2, metric="euclidean"):
    if len(x1) != len(x2):
        raise ValueError("The dimensions of the two points must be the same.")

    if metric == "euclidean":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

    elif metric == "cosine":
        dot_product = sum(a * b for a, b in zip(x1, x2))
        magnitude_x1 = math.sqrt(sum(a ** 2 for a in x1))
        magnitude_x2 = math.sqrt(sum(b ** 2 for b in x2))
        return 1 - (dot_product / (magnitude_x1 * magnitude_x2))        

    elif metric == "manhattan":
        return sum(abs(a - b) for a, b in zip(x1, x2))        

    elif metric == "chebyshev":
        return max(abs(a - b) for a, b in zip(x1, x2))

    raise ValueError(f"Distance metric {metric} is not supported.")


def kmeans(data, k, metric="euclidean", max_iters=100):
    centroids = random.sample(data, k)
    
    for _ in range(max_iters):
        clusters = defaultdict(list)
        
        for point in data:
            distances = [distance(point, centroid, metric) for centroid in centroids]
            nearest_centroid_idx = distances.index(min(distances))
            clusters[nearest_centroid_idx].append(point)
        
        new_centroids = []
        for cluster_points in clusters.values():
            cluster_mean = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
            new_centroids.append(cluster_mean)
        
        if centroids == new_centroids:
            break
        
        centroids = new_centroids
    
    return centroids, clusters


# Example usage:
if __name__ == "__main__":
    # Sample data
    data = [[1, 2], [1, 1], [2, 3], [8, 7], [9, 8], [7, 9]]
    
    # Number of clusters
    k = 2
    
    # Metrics
    metrics = ["euclidean", "cosine", "manhattan", "chebyshev",]

    # Run K-means with all distance metrics
    for metric in metrics:
        centroids, clusters = kmeans(data, k, metric=metric)
        print(f"Centroids ({metric}):", centroids)
        print(f"Clusters ({metric}):", dict(clusters))
        print()
