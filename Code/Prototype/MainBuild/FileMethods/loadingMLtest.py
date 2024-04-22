import time

import cv2
import numpy as np

from Code.Prototype.MainBuild.FileMethods.GenerateMLmodel import loadValues

query_data, query_labels = loadValues('TrainingData/query_dataTemp.arff')

#print(query_data)
tic = time.perf_counter()
ML_model = cv2.ml.RTrees_load("MLmodels/random_forest_model.xml")

first_array = query_data[0]
array = first_array.tolist()
final_array = [array]
print(np.array(final_array, dtype=np.float32))

tic = time.perf_counter()
_, predictions = ML_model.predict(np.array(final_array, dtype=np.float32))
print(predictions[0][0])

if predictions[0][0] == 2.0:
    print("true")
"""extracted_values = np.array([x[0] for x in predictions])
temp_array = np.empty([1, 1])
correct_predictions = np.sum(extracted_values == query_labels)
print(correct_predictions)
total_samples = len(query_labels)
accuracy = (correct_predictions / total_samples) * 100
print(str(accuracy) + '%')"""

toc = time.perf_counter()
print(f"completed all classification in {toc - tic:0.4f} seconds")


#<0.0001 seconds to classify one set of values.

#0.0003 seconds to classify one set of values with increased random tree depth

# #longer to extract though