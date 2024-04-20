import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np
from matplotlib.widgets import CheckButtons

arff_file_path = '../MainBuild/FileMethods/TrainingData/training_dataTemp.arff'
data, meta = arff.loadarff(arff_file_path)
hue_values = data['circularity2']
class_labels = data['class']

# Replace x and y with your desired range for the random numbers
groats_hues = []
wholegrain_hues = []
broken_hues = []


for hue, label in zip(hue_values, class_labels):
    if label == b'wholegrain':  # Adjust the class names based on your actual class names
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
# Generate a large set of random numbers
wholegrain_hist, _, _ = ax.hist(wholegrain_hues, bins=1000, color='orange', alpha=0.5, label='Wholegrain')
broken_hist, _, _ = ax.hist(broken_hues, bins=1000, color='blue', alpha=0.5, label='Broken')
groats_hist, _, _ = ax.hist(groats_hues, bins=1000, color='green', alpha=0.5, label='Groats')
# Add labels and title
plt.xlabel('circularity2')
plt.ylabel('Frequency')
plt.title('circularity2 Distribution')

legend = leg = ax.legend(fancybox=True, shadow=True)

plt.show()