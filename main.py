import random
from math import exp

# create matrix of size f*s, initialized with randomFloat() entries
# with f, s sizes of first and second layer


def createRandomWeights(firstlayer, secondlayer):
    return [[randomFloat() for x in range(firstlayer.size)]
            for y in range(secondlayer.size)]

# normalize x to be a value between 0 and 1


def normalize(x):
    return -1 + 2 / (1 + exp(- x)) if x > 0 else 0

# create a random float number between -5 and 5


def randomFloat(lower=-5, upper=5):
    return random.uniform(lower, upper)

# layer class, input size
# creates value list of that size
# creates bias list of that size, initialized with randomFloat() entries


class LAYER():
    def __init__(self, size):
        self.size = size
        self.values = [0 for _ in range(self.size)]
        self.biases = [randomFloat() for _ in range(self.size)]

    # for each value of layer, sum products of weight and input of the
    # corresponsing input values and weightMatrix weights
    # change layer values to be the normalized sum
    def propagate(self, incomingLayer, weightMatrix):
        for i in range(self.size):
            current = 0
            for j in range(incomingLayer.size):
                current += incomingLayer.values[j] * weightMatrix[i][j]
            self.values[i] = normalize(self.biases[i] + current)

# neural net class
# initializes 5 layers, creates the random weights between them


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

    # feeds incoming data through the net, calculating the values of the
    # following layer and ultimately giving the index of the output
    # neuron with the largest value
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
