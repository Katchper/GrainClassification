"""
Demo4
Author: Kacper Dziedzic ktd1
Version: 1.1

Demonstration showing a histogram of the hues of the 3 types of grain.

"""

import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np


arff_file_path = '../MainBuild/FileMethods/TrainingData/training_dataTemp3.arff'
data, meta = arff.loadarff(arff_file_path)
hue_values = data['hue']
class_labels = data['class']

groats_hues = []
wholegrain_hues = []
broken_hues = []


for hue, label in zip(hue_values, class_labels):
    if label == b'wholegrain':
        wholegrain_hues.append(hue)
    elif label == b'broken':
        broken_hues.append(hue)
    elif label == b'groats':
        groats_hues.append(hue)

wholegrain_hues = np.array(wholegrain_hues)
broken_hues = np.array(broken_hues)
groats_hues = np.array(groats_hues)

fig, ax = plt.subplots()
wholegrain_hist, _, _ = ax.hist(wholegrain_hues, bins=200, color='orange', alpha=0.5, label='Wholegrain')
broken_hist, _, _ = ax.hist(broken_hues, bins=200, color='blue', alpha=0.5, label='Broken')
groats_hist, _, _ = ax.hist(groats_hues, bins=200, color='green', alpha=0.5, label='Groats')

custom_ticks_x = np.linspace(0, 1, 21)
plt.xticks(custom_ticks_x)

plt.xlabel('Rectangularity')
plt.ylabel('Frequency')
plt.title('Hue Distribution')

legend = leg = ax.legend(fancybox=True, shadow=True)

lines = [wholegrain_hist, broken_hist, groats_hist]
map_legend_to_ax = {}
pickradius = 5

for legend_line, ax_line in zip(leg.get_lines(), lines):
    legend_line.set_picker(pickradius)
    map_legend_to_ax[legend_line] = ax_line

plt.show()