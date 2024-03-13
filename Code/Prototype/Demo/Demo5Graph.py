import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np
from matplotlib.widgets import CheckButtons

arff_file_path = '../MainBuild/FileMethods/TrainingData/training_dataTemp.arff'
data, meta = arff.loadarff(arff_file_path)
hue_values = data['perimeter']
class_labels = data['class']

# Replace x and y with your desired range for the random numbers
groats_hues = []
wholegrain_hues = []
broken_hues = []


for hue, label in zip(hue_values, class_labels):
    if label == b'wholegrain':  # Adjust the class names based on your actual class names
        wholegrain_hues.append(hue)
    elif label == b'broken':
        broken_hues.append(hue)
    elif label == b'groats':
        groats_hues.append(hue)

wholegrain_hues = np.array(wholegrain_hues)
broken_hues = np.array(broken_hues)
groats_hues = np.array(groats_hues)

fig, ax = plt.subplots()
# Generate a large set of random numbers
wholegrain_hist, _, _ = ax.hist(wholegrain_hues, bins=2000, color='orange', alpha=0.5, label='Wholegrain')
broken_hist, _, _ = ax.hist(broken_hues, bins=2000, color='blue', alpha=0.5, label='Broken')
groats_hist, _, _ = ax.hist(groats_hues, bins=2000, color='green', alpha=0.5, label='Groats')
custom_ticks_x = np.linspace(0, 10000, 20)
plt.xticks(custom_ticks_x)
# Add labels and title
plt.xlabel('Perimeter')
plt.ylabel('Frequency')
plt.title('Perimeter Distribution')

legend = leg = ax.legend(fancybox=True, shadow=True)
ax.set_xlim(0, 500)

plt.show()