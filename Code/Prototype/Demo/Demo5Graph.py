"""
Demo 5
Author: Kacper Dziedzic ktd1
Version: 1.1


aspect ratio graph, displays the histogram for aspect ratio.
This can be replaced by any other attribute for visualisation purposes.
"""

import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np

arff_file_path = '../MainBuild/FileMethods/TrainingData/training_dataTemp.arff'
data, meta = arff.loadarff(arff_file_path)
aspectRatio_values = data['aspectRatio']
class_labels = data['class']

groats_aspectRatio = []
wholegrain_aspectRatio = []
broken_aspectRatio = []

# sort the values into 3 seperate lists
# convert arff into variables
for aspectRatio, label in zip(aspectRatio_values, class_labels):
    if label == b'wholegrain':
        wholegrain_aspectRatio.append(aspectRatio)
    elif label == b'broken':
        broken_aspectRatio.append(aspectRatio)
    elif label == b'groats':
        groats_aspectRatio.append(aspectRatio)

wholegrain_aspectRatio = np.array(wholegrain_aspectRatio)
broken_aspectRatio = np.array(broken_aspectRatio)
groats_aspectRatio = np.array(groats_aspectRatio)

fig, ax = plt.subplots()
# Bins is how many bars are used for the histogram
wholegrain_hist, _, _ = ax.hist(wholegrain_aspectRatio, bins=300, color='orange', alpha=0.5, label='Wholegrain')
broken_hist, _, _ = ax.hist(broken_aspectRatio, bins=300, color='blue', alpha=0.5, label='Broken')
groats_hist, _, _ = ax.hist(groats_aspectRatio, bins=300, color='green', alpha=0.5, label='Groats')

# Add labels and title to graph
plt.xlabel('aspectRatio')
plt.ylabel('Frequency')
plt.title('aspectRatio Distribution')
legend = leg = ax.legend(fancybox=True, shadow=True)
# limit axis to between 0-1
ax.set_xlim(0, 1)

plt.show()