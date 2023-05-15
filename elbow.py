import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


def calculate_inertia(vectors, max_clusters=12):
    """
    Calculate the inertia values for a range of clusters.

    :param vectors: A list of vectors to cluster.

    :param max_clusters: The maximum number of clusters to use.

    :return: A list of inertia values.
    """
    inertia_values = []
    for num_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        inertia_values.append(kmeans.inertia_)
    return inertia_values


def plot_elbow(inertia_values):
    """
    Plot the inertia values for a range of clusters. Just for fun!

    :param inertia_values: A list of inertia values.

    :return: None.
    """
    plt.plot(inertia_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()


def determine_optimal_clusters(inertia_values):
    """
    Determine the optimal number of clusters to use based on the inertia values.

    :param inertia_values: A list of inertia values.

    :return: The optimal number of clusters to use.
    """
    distances = []
    for i in range(len(inertia_values) - 1):
        p1 = np.array([i + 1, inertia_values[i]])
        p2 = np.array([i + 2, inertia_values[i + 1]])
        d = np.linalg.norm(np.cross(p2 - p1, p1 - np.array([1,0]))) / np.linalg.norm(p2 - p1)
        distances.append(d)
    optimal_clusters = distances.index(max(distances)) + 2
    return optimal_clusters

