import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier

#from Code.Prototype.MainBuild.FileMethods.loadRandomData import generateQueryFile

#print("generating random query file")
#generateQueryFile()
#print("done")

def startJvm():
    jvm.start(max_heap_size="8192m")

def machineLearningAlgorithm(training, query):
    # the memory allocated to the machine learning model (more usually makes it faster)

    data_dir = training
    query_dir = query

    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_dir)
    data.class_is_last()

    test = loader.load_file(query_dir)
    test.class_is_last()
    #print("training dataset")
    cls = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-P", "100", "-I", "100", "-num-slots", "1", "-K", "0", "-M", "1.0", "-V", "0.001", "-S", "1", "-num-decimal-places", "5"])

    #print(cls.options)
    cls.build_classifier(data)
    #print(cls)
    #print("done")
    # output from testing the query file stats
    #print("# - actual - predicted - error - distribution")
    correct = 1
    total = 1
    totalWhole = 1
    totalBroken = 1
    totalGroat = 1
    detectedWhole = 1
    detectedBroken = 1
    detectedGroat = 1

    prediction_list = []
    for index, inst in enumerate(test):
        total += 1
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        prediction_list.append(inst.class_attribute.value(int(pred)))

        if inst.get_string_value(inst.class_index) == "wholegrain":
            totalWhole += 1
            if pred == inst.get_value(inst.class_index):
                detectedWhole += 1
        elif inst.get_string_value(inst.class_index) == "broken":
            totalBroken += 1
            if pred == inst.get_value(inst.class_index):
                detectedBroken += 1
        else:
            totalGroat += 1
            if pred == inst.get_value(inst.class_index):
                detectedGroat += 1

        #print(
        #    "%d - %s - %s - %s  - %s" %
        #    (index+1,
        #    inst.get_string_value(inst.class_index),
        #    inst.class_attribute.value(int(pred)),
         #   "yes" if pred != inst.get_value(inst.class_index) else "no",
        #    str(dist.tolist())))
        if pred == inst.get_value(inst.class_index):
            correct += 1
    accuracy = (correct/total) * 100
    #print("accuracy this iteration: %" + str(accuracy))
    print("total wholewheat: " + str(totalWhole-1) + " detected wholewheat: " + str(detectedWhole-1) + " accuracy: " + str((detectedWhole/totalWhole) * 100) + "%")
    print("total groats: " + str(totalGroat-1) + " detected groats: " + str(detectedGroat-1) + " accuracy: " + str((detectedGroat / totalGroat) * 100) + "%")
    print("total broken: " + str(totalBroken-1) + " detected broken: " + str(detectedBroken-1) + " accuracy: " + str((detectedBroken / totalBroken) * 100) + "%")
    return prediction_list

def stopJvm():
    jvm.stop()

"""startJvm()
machineLearningAlgorithm("FileMethods/TrainingData/training_dataTemp.arff","FileMethods/TrainingData/query_dataTemp.arff")
stopJvm()
"""
