import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier

#from Code.Prototype.MainBuild.FileMethods.loadRandomData import generateQueryFile

#print("generating random query file")
#generateQueryFile()
#print("done")
def machineLearningAlgorithm():
    # the memory allocated to the machine learning model (more usually makes it faster)
    jvm.start(max_heap_size="8192m")

    data_dir = "TrainingData/training_data.arff"
    query_dir = "TrainingData/query.arff"

    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(data_dir)
    data.class_is_last()

    test = loader.load_file(query_dir)
    test.class_is_last()
    print("training dataset")
    cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
    print(cls.options)
    cls.build_classifier(data)
    print(cls)
    print("done")
    # output from testing the query file stats
    print("# - actual - predicted - error - distribution")
    correct = 0
    total = 0
    prediction_list = []
    for index, inst in enumerate(test):
        total += 1
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        prediction_list.append(inst.class_attribute.value(int(pred)))
        print(
            "%d - %s - %s - %s  - %s" %
            (index+1,
            inst.get_string_value(inst.class_index),
            inst.class_attribute.value(int(pred)),
            "yes" if pred != inst.get_value(inst.class_index) else "no",
            str(dist.tolist())))
        if pred == inst.get_value(inst.class_index):
            correct += 1
    accuracy = (correct/total) * 100
    print("accuracy: %" + str(accuracy))

    jvm.stop()
    return prediction_list
