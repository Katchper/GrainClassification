"""
Demo 6
Author: Kacper Dziedzic ktd1
Version: 1.1

Demonstration showing the hu moment 1 distribution. Other values can easily be replaced
"""

import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np

arff_file_path = '../MainBuild/FileMethods/TrainingData/training_dataTemp.arff'
data, meta = arff.loadarff(arff_file_path)
# change text here to switch the attribute
hue_values = data['hueMoment1']
class_labels = data['class']

groats_hues = []
wholegrain_hues = []
broken_hues = []


for hue, label in zip(hue_values, class_labels):
    if label == b'wholegrain':
        if len(wholegrain_hues) < 9000:
            wholegrain_hues.append(hue)
    elif label == b'broken':
        if len(broken_hues) < 9000:
            broken_hues.append(hue)
    elif label == b'groats':
        if len(groats_hues) < 9000:
            groats_hues.append(hue)

wholegrain_hues = np.array(wholegrain_hues)
broken_hues = np.array(broken_hues)
groats_hues = np.array(groats_hues)

print(len(groats_hues))
print(len(wholegrain_hues))
print(len(broken_hues))

fig, ax = plt.subplots()
# change the bins to get a more or less blocky histogram
wholegrain_hist, _, _ = ax.hist(wholegrain_hues, bins=1000, color='orange', alpha=0.5, label='Wholegrain')
broken_hist, _, _ = ax.hist(broken_hues, bins=1000, color='blue', alpha=0.5, label='Broken')
groats_hist, _, _ = ax.hist(groats_hues, bins=1000, color='green', alpha=0.5, label='Groats')

# Add labels and title (change titles when changing feature)
plt.xlabel('huMoment1')
plt.ylabel('Frequency')
plt.title('huMoment1 Distribution')

legend = leg = ax.legend(fancybox=True, shadow=True)

plt.show()