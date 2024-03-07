import random


def random_line(fname):
    lines = open(fname).read().splitlines()
    return random.choice(lines)


def generateQueryFile():
    file_name = "TrainingData/query.arff"
    data_file = "TrainingData/training_data.arff"
    file = open(file_name, "w")
    file.write("@RELATION ImageDataset\n")
    file.write("\n")
    file.write("@ATTRIBUTE hue NUMERIC\n")
    file.write("@ATTRIBUTE saturation NUMERIC\n")
    file.write("@ATTRIBUTE value NUMERIC\n")
    file.write("@ATTRIBUTE area NUMERIC\n")
    file.write("@ATTRIBUTE class {wholegrain, groats, broken}\n")
    file.write("\n")
    file.write("@DATA\n")

    # here is where the data goes in the format: red, blue, green, area, grainType
    # example is: file.write("45, 32, 67, 1200, groat\n")

    for x in range(200):
        with open(data_file) as f:
            lines = f.readlines()
            file.write(random.choice(lines))
    file.close()


generateQueryFile()
