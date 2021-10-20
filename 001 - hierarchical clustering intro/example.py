import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift


# Blob parameters
NUMBER_OF_SAMPLES = 200
CLUSTER_CENTERS = [(1,1), (4,4)]
CLUSTER_STD = 1

# Additional parameters
list_of_colors = ["r", "b", "c", "g"]

# Generate data
data_points, labels_data_points = make_blobs(   n_samples=NUMBER_OF_SAMPLES, 
                                                centers=CLUSTER_CENTERS,
                                                cluster_std=CLUSTER_STD )

# Plot and save generated Blob data
fig, axs = plt.subplots()
axs.scatter(data_points[:,0], data_points[:,1])
fig.savefig("./figures/generated_data_points.png")

# Fit data
mean_shift = MeanShift().fit(data_points)
calculated_labels = mean_shift.labels_
calculated_cluster_centers = mean_shift.cluster_centers_
calculated_number_of_clusters = len(np.unique(calculated_labels))

# Plot and save calculated classification data
fig, axs = plt.subplots()
for i, data_point in enumerate(data_points):
    axs.scatter(data_point[0], data_point[1], c=list_of_colors[calculated_labels[i]])

axs.scatter(calculated_cluster_centers[:,0], calculated_cluster_centers[:,1], marker="x", s=200, c="k")
fig.savefig("./figures/calculated_data_points.png")







