import numpy as np
import random
import math
from matplotlib import pyplot as plt, colors as mcolors
from sklearn.datasets import make_blobs

# Blob parameters
NUMBER_OF_SAMPLES = 500
NUMBER_OF_CLUSTERS = 5
CLUSTER_CENTERS = [(random.random(), random.random()) for _ in range(NUMBER_OF_CLUSTERS)]
CLUSTER_STD = 0.05

# Generate data
data_points, labels_data_points = make_blobs(   n_samples=NUMBER_OF_SAMPLES, 
                                                centers=CLUSTER_CENTERS,
                                                cluster_std=CLUSTER_STD )

number_of_k_mean = NUMBER_OF_CLUSTERS
max_number_of_iterations = 10

list_colors = math.ceil(number_of_k_mean/5)*["b", "k", "g", "y", "m"]

def normalise_data_points(data_points):
    get_max_min = lambda x: [min(x), max(x)-min(x)]

    min_x_value, diff_max_min_x = get_max_min([data_point[0] for data_point in data_points])
    min_y_value, diff_max_min_y = get_max_min([data_point[1] for data_point in data_points])

    return np.array([ ((data_point[0]-min_x_value)/diff_max_min_x, (data_point[1]-min_y_value)/diff_max_min_y) 
                        for data_point in data_points])
                
def get_uniform_distributed_k_means(number_of_k_mean):
    return np.array([[random.random(), random.random()] for _ in range(number_of_k_mean)])

def get_euclidean_distance_to_k_means(point, k_means):
    return np.array([np.linalg.norm(point-k_means[i,:]) for i in range(k_means.shape[0])])

def main():
    normalised_data_points = normalise_data_points(data_points=data_points)
    k_means = get_uniform_distributed_k_means(number_of_k_mean)

    argmin_distances = np.zeros(NUMBER_OF_SAMPLES)

    for i in range(max_number_of_iterations):
        
        print("Iteration %i" % i)

        list_of_cluster_indices = []

        for j in range(NUMBER_OF_SAMPLES):
            euclidean_distances = get_euclidean_distance_to_k_means(normalised_data_points[j,:], k_means)
            argmin_distances[j] = np.argmin(euclidean_distances)

        for k in range(len(k_means)):
            cluster_indices = [i for i, argmin_distance in enumerate(argmin_distances) if argmin_distance == k]
            if cluster_indices: k_means[k,:] = np.mean(normalised_data_points[cluster_indices,:], axis=0)
            list_of_cluster_indices.append(cluster_indices)

        fig, axs = plt.subplots()

        for j, cluster_indices in enumerate(list_of_cluster_indices):
            axs.scatter(normalised_data_points[cluster_indices,0], normalised_data_points[cluster_indices,1], c=list_colors[j])
            if cluster_indices:
                axs.scatter(k_means[j,0], k_means[j,1], c="r")
        axs.set_xlim([0, 1]); axs.set_ylim([0, 1])

        plt.savefig(f"figures/results_iter_{i}.png")
        plt.close()
    

if __name__ == "__main__":
    main()




