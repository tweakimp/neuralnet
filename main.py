import random
from math import exp


def createRandomWeights(firstlayer, secondlayer):
    return [[randomFloat() for x in range(firstlayer.size)]
            for y in range(secondlayer.size)]


def createRandomBiases(layer):
    return [randomFloat() for x in layer]


def normalize(x):
    return -1 + 2 / (1 + exp(- x)) if x > 0 else 0


def randomFloat(lower=-5, upper=5):
    return random.uniform(lower, upper)


class LAYER():
    def __init__(self, size):
        self.size = size
        self.values = [0 for _ in range(self.size)]
        self.biases = [randomFloat() for _ in range(self.size)]

    def propagate(self, incomingLayer, weightMatrix):
        for i in range(self.size):
            current = 0
            for j in range(incomingLayer.size):
                current += incomingLayer.values[j]*weightMatrix[i][j]
            self.values[i] = normalize(self.biases[i]+current)


class NEURALNET():

    def __init__(self):
        self.sizes = {"input": 49, "hidden1": 49, "hidden2": 49,
                      "hidden3": 49, "output": 9}

        self.inputlayer = LAYER(self.sizes["input"])
        self.hidden1 = LAYER(self.sizes["hidden1"])
        self.hidden2 = LAYER(self.sizes["hidden2"])
        self.hidden3 = LAYER(self.sizes["hidden3"])
        self.outputlayer = LAYER(self.sizes["output"])

        self.in_h1_weights = createRandomWeights(self.inputlayer, self.hidden1)
        self.h1_h2_weights = createRandomWeights(self.hidden1, self.hidden2)
        self.h2_h3_weights = createRandomWeights(self.hidden2, self.hidden3)
        self.h3_out_weights = createRandomWeights(
            self.hidden3, self.outputlayer)

    def feed(self, inputdata):
        # input data
        for x in range(self.inputlayer.size):
            self.inputlayer.values[x] = inputdata[x]
        # input to h1
        self.hidden1.propagate(self.inputlayer, self.in_h1_weights)
        # h1 to h2
        self.hidden2.propagate(self.hidden1, self.h1_h2_weights)
        # h2 to h3
        self.hidden3.propagate(self.hidden2, self.h2_h3_weights)
        # h3 to output
        self.outputlayer.propagate(self.hidden3, self.h3_out_weights)
        # output
        return self.outputlayer.values.index(max(self.outputlayer.values))
