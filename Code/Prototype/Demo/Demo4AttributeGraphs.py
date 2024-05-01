import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np
from matplotlib.widgets import CheckButtons

arff_file_path = '../MainBuild/FileMethods/TrainingData/training_dataTemp3.arff'
data, meta = arff.loadarff(arff_file_path)
hue_values = data['rectangularity']
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
wholegrain_hist, _, _ = ax.hist(wholegrain_hues, bins=200, color='orange', alpha=0.5, label='Wholegrain')
broken_hist, _, _ = ax.hist(broken_hues, bins=200, color='blue', alpha=0.5, label='Broken')
groats_hist, _, _ = ax.hist(groats_hues, bins=200, color='green', alpha=0.5, label='Groats')

custom_ticks_x = np.linspace(0, 1, 21)
plt.xticks(custom_ticks_x)
# Add labels and title
plt.xlabel('Hue')
plt.ylabel('Frequency')
plt.title('Hue Distribution')

legend = leg = ax.legend(fancybox=True, shadow=True)

lines = [wholegrain_hist, broken_hist, groats_hist]
map_legend_to_ax = {}
pickradius = 5

for legend_line, ax_line in zip(leg.get_lines(), lines):
    legend_line.set_picker(pickradius)  # Enable picking on the legend line.
    map_legend_to_ax[legend_line] = ax_line

def on_pick(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    legend_line = event.artist

    # Do nothing if the source of the event is not a legend line.
    if legend_line not in map_legend_to_ax:
        return

    ax_line = map_legend_to_ax[legend_line]
    visible = not ax_line.get_visible()
    ax_line.set_visible(visible)
    # Change the alpha on the line in the legend, so we can see what lines
    # have been toggled.
    legend_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()