import numpy as np
import math
import time

#information about network to be trained
numOfPixles = 5 #change this value to for different number of pixles
numOfNodes = [numOfPixles, 10, 20, 1] #length must match number of layers
numOfLayers = 4 #change this for different number of layers in neural network
learningrate = 0.5 #change this value to receive different leaning rates

#loging
logging = False;
def log(message):
    if logging:
        print(message)

#data parsing
def openImage(path):
    pass

#neural network
def stigmoid(x):
    return 1/(1 + math.exp(-1 * x))
vector_stigmoid = np.vectorize(stigmoid)

#init weights and biases
weights = [(np.random.rand(numOfNodes[x], numOfNodes[x - 1]) - 0.5) * 12 for x in range(1, numOfLayers)]
biases = [(np.random.rand(numOfNodes[x]) - 0.5) * 12 for x in range(1, numOfLayers)]

def runNeuralNetwork(input): #give pixle values in 32-bit floats. Length must mach numOfPixles
    nodeValues = [np.array(input)]
    for x in range(0, numOfLayers - 1):
        nodeValues.append(vector_stigmoid(np.matmul(weights[x], nodeValues[x]) + biases[x]))
    return nodeValues

#backpropagation

def cost(out, expected):
    return math.pow(expected - out, 2)
vector_cost = np.vectorize(cost)

def d_stigmoid(x):
    return 1/(math.exp(x) * math.pow(1 + math.exp(-x), 2))
vector_d_stigmoid = np.vectorize(d_stigmoid)

def r_stigmoid(x):
    return math.log(x/(1 - x));
vector_r_stigmoid = np.vectorize(r_stigmoid)

def d_cost(out, expected):
    return 2 * (expected - out)
vector_d_cost = np.vectorize(d_cost)

def backpropagation(nodeValues, expectedValues):
    gradient = g_weights, g_biases = [], []
    g_node = np.array(vector_d_cost(nodeValues[numOfLayers - 1], expectedValues))

    for x in range(0, numOfLayers - 1):
        g_node *= vector_d_stigmoid(vector_r_stigmoid(nodeValues[-1 * (x + 1)]))
        g_biases.insert(0, np.array(g_node))
        g_weights.insert(0,
                         np.transpose(np.resize(g_node, (numOfNodes[-1 * (x + 2)], numOfNodes[-1 * (x + 1)]))) *
                         np.resize(nodeValues[-1 * (x + 1)], (numOfNodes[-1 * (x + 1)], numOfNodes[-1 * (x + 2)]))
                         )
        g_node = np.matmul(np.transpose(weights[-1 * (x + 1)]), g_node)

    return gradient

def applyGradient(weights_gradient, bias_gradient):
    for x in range(0, numOfLayers - 1):
        weights[x] +=  weights_gradient[x] * learningrate
        biases[x] += bias_gradient[x] * learningrate

#script
logging = True

for x in range(0, 10000):
    output = runNeuralNetwork([0.3, 0.2, 0.5, 0.2, 0.8])
    gradientVector = backpropagation(output, np.array([.5]))
    for y in output[3].tolist():
        print(str(x) + ": Value of output = "+ str(y))
    log(str(x) + ": Cost of run: " + str(cost(output[3].tolist()[0], 0.5)))
    log("_____________________________________")
    applyGradient(gradientVector[0], gradientVector[1])


