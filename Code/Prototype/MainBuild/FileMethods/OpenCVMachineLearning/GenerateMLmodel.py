"""
GenerateMLmodel
Author: Kacper Dziedzic ktd1
Version: 1.1

Methods to create a random trees machine learning model within OpenCV
This methods use is exclusively to create and upload an ML model to the JeVois camera
"""

import cv2
import numpy as np
from scipy.io import arff
def loadValues(dir):
    arff_file_path = dir
    data, meta = arff.loadarff(arff_file_path)
    hue1 = data['hue']
    sat1 = data['saturation']
    val1 = data['value']
    hu11 = data['hueMoment1']
    hu21 = data['hueMoment2']
    hu31 = data['hueMoment3']
    hu41 = data['hueMoment4']
    hu51 = data['hueMoment5']
    hu61 = data['hueMoment6']
    hu71 = data['hueMoment7']
    circ1 = data['circularity']
    circ21 = data['circularity2']
    rect1 = data['rectangularity']
    aspect1 = data['aspectRatio']
    compact1 = data['compactness']
    labels = np.array(data['class'])
    # convert labels into numbers
    final_features = []
    final_labels = []
    for label, hue, sat, val, hu1, hu2, hu3, hu4, hu5, hu6, hu7, circ, circ2, rect, aspect, compact in zip(labels, hue1, sat1, val1, hu11, hu21, hu31, hu41, hu51, hu61, hu71, circ1, circ21, rect1, aspect1, compact1):
        final_features.append([hue, sat, val, hu1, hu2, hu3, hu4, hu5, hu6, hu7, circ, circ2, rect, aspect, compact])
        if label == b'wholegrain':
            temp = 2
            final_labels.append(temp)
        elif label == b'groats':
            temp = 1
            final_labels.append(temp)
        else:
            temp = 0
            final_labels.append(temp)
    return np.array(final_features, dtype=np.float32), np.array(final_labels, dtype=np.int32)


def saveModel():
    training_data, training_labels = loadValues('../TrainingData/training_dataTemp.arff')
    query_data, query_labels = loadValues('../TrainingData/query_dataTemp.arff')
    print(len(query_data))
    print(len(query_labels))

    ML_temp = cv2.ml.RTrees_create()

    # Set parameters
    ML_temp.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 200, 0.01))
    ML_temp.setMinSampleCount(2)
    ML_temp.setMaxDepth(10)
    ML_temp.setMaxCategories(15)

    ML_temp.train(training_data, cv2.ml.ROW_SAMPLE, training_labels)

    _, predictions = ML_temp.predict(query_data)

    extracted_values = np.array([x[0] for x in predictions])

    correct_predictions = np.sum(extracted_values == query_labels)
    print(correct_predictions)
    total_samples = len(query_labels)
    accuracy = (correct_predictions / total_samples) * 100
    print(accuracy)

    ML_temp.save("MLmodels/random_trees_model.xml")

saveModel()