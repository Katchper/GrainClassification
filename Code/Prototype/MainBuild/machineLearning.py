from weka.classifiers import IBk
c = IBk(K=1)
c.train("TrainingData/training_data.arff")
predictions = c.predict("TrainingData/query.arff")