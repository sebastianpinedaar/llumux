import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

def kmedoids(X, k, max_iter=300):
    n_samples = len(X)
    
    # Step 1: Randomly choose k medoids (indices)
    medoid_indices = np.random.choice(n_samples, size=k, replace=False)
    medoids = X[medoid_indices]

    for _ in range(max_iter):
        # Step 2: Assign points to the nearest medoid
        distances = cdist(X, medoids)
        labels = np.argmin(distances, axis=1)
        
        # Step 3: Update medoids
        new_medoids = []
        new_medoids_indices = []
        for cluster_idx in range(k):
            cluster_points = X[labels == cluster_idx]
            if len(cluster_points) == 0:
                continue  # skip empty cluster
            # Compute all pairwise distances in the cluster
            intra_distances = cdist(cluster_points, cluster_points)
            total_distances = np.sum(intra_distances, axis=1)
            best_idx = np.argmin(total_distances)
            new_medoids.append(cluster_points[best_idx])
            new_medoids_indices.append(np.where(labels == cluster_idx)[0][best_idx])
        new_medoids = np.array(new_medoids)
        new_medoids_indices = np.array(new_medoids_indices)
        # Convergence check
        if np.allclose(medoids, new_medoids):
            break
        medoids = new_medoids
        medoid_indices = new_medoids_indices

    # Final labels
    final_distances = cdist(X, medoids)
    final_labels = np.argmin(final_distances, axis=1)
    
    return final_labels, medoids, medoid_indices