"""
Demo 3
Author: Kacper Dziedzic ktd1
Version: 1.1

Matplotlib visualisation line graph showing the statistics I got from initial ML model tests.
"""

import matplotlib.pyplot as plt
import numpy as np
# manually set points
xpoints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#ypoints = np.array([93.99, 94.33, 94.35, 94.20, 94.35, 94.23, 94.34, 94.74, 94.23, 94.36, 94.17, 94.91, 94.14, 94.16, 94.39, 94.57, 93.72, 94.85, 94.47, 94.26])

ypointsW = np.array([
    90.61338289962825,
    90.71686436307374,
    95.55,
    98.25,
    96.98,
    95.96,
    98.57,
    95.85,
    98.43,
    98.8
])
ypointsG = np.array([
    96.5382974183913,
    97.43868026915563,
    99.1,
    96.81,
    94.17,
    94.52,
    95.99,
    90.75,
    93.83,
    96.4
])

ypointsB = np.array([
    73.54577642893273,
    67.76071995764956,
    70.93,
    92.1,
    86.21,
    85.73,
    85.87,
    94.21,
    92.83,
    92.6
])
#ypointsAccuracy = np.array([86.8991522489841, 85.30542152995963, 85.03002998414827, 85.8457510380278, 86.80637688576952, 85.45348760134068, 85.2036028617877, 85.16572236860183, 85.02679137625957, 85.51518273165907, 85.2812246117064, 86.95462161132532, 84.08948129342791, 85.43587869116091, 84.34777593004084, 86.30831481200674, 84.22088068969629, 84.62133978208261, 85.58627882396927, 84.4207292259071])

ypointsAccuracy = np.array([86.565, 85.642, 88.196, 95.738, 92.792, 92.073, 93.811, 93.294, 95.033, 96.27])

#Create the lines
plt.plot(xpoints, ypointsW, label='Wholewheat Accuracy', linewidth=1)
plt.plot(xpoints, ypointsG, label='Groats Accuracy', linewidth=1)
plt.plot(xpoints, ypointsB, label='Broken Accuracy', linewidth=1)
#plt.plot(xpoints, ypoints, label='Total Accuracy', linewidth=2.5)
plt.plot(xpoints, ypointsAccuracy, label='Average Accuracy', linewidth=2)

custom_ticks_x = np.linspace(1, 10, 10)
custom_ticks_y = np.linspace(0, 100, 41)  # Adjust the number of ticks as needed
plt.xticks(custom_ticks_x)
plt.yticks(custom_ticks_y)

plt.xlabel('Iteration')
plt.ylabel('Accuracy %')
plt.title('Accuracy statistics')

plt.legend()
plt.show()