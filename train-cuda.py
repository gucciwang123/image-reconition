import numpy as np
from numba import vectorize, jit, float64, cuda
import struct
import math
import cupy as cp

#information about network to be trained
numOfPixles = 784 #change this value to for different number of pixles
numOfNodes = [numOfPixles, 16, 16, 10] #length must match number of layers
numOfLayers = 4 #change this for different number of layers in neural network
learningrate = 1.0E-1 #change this value to receive different leaning rates
runsPerEvolution =1000 #change this for more runs for every evolution of the neural network
numOfEvolutions = 60 #change this for number of iterations to run
readFile = False #change to true if reading in a neural network
networkName = "cudatest"
numOfIterations = 20 #change for the training run to be iterated a differnent amount of times

#network configuration
if readFile:
    weights = [np.load(networkName + ".w." + str(x) + ".npy", allow_pickle=True) for x in range(numOfLayers - 1)]
    biases = [np.load(networkName + ".b." + str(x) + ".npy", allow_pickle=True) for x in range(numOfLayers - 1)]
else:
    weights = [(np.random.rand(numOfNodes[x], numOfNodes[x - 1]) - 0.5) * 6 for x in range(1, numOfLayers)]
    biases = [(np.random.rand(numOfNodes[x]) - 0.5) * 6 for x in range(1, numOfLayers)]

#math functions
@vectorize(['float64(float64, float64)'], target="cuda")
def cost(expected, outcome):
    return pow((expected - outcome), 2)

@vectorize(['float64(float64, float64)'], target="cuda")
def d_cost(out, expected):
    return 2 * (expected - out)

@vectorize(['float64(float64)'], target="cuda")
def stigmoid(x):
    return 1/(1 + math.exp(-1 * x))

@vectorize(['float64(float64)'], target="cuda")
def d_stigmoid(x):
    return 1/(1 + math.exp(-1 * x)) * (1 - 1/(1 + math.exp(-1 * x)))

@vectorize(['float64(float64)'], target="cuda")
def r_stigmoid(x):
    if x >= 1:
        return 1
    if x <= 0:
        return 0
    return math.log(x/(1 - x))

@jit()
def runNetwork(x): #input a numpy vector of initial node activations; returns array of numpy vectors with each node activation
    activations = [np.array(x)]

    for i in range(1, numOfLayers):
        activations.append(
            stigmoid((np.matmul(weights[i - 1], activations[i -1]) + biases[i - 1]))
        )

    return activations

@jit()
def backprogation(nodeValues, expectedValues):
    g_weights, g_biases = [], []
    g_node = np.array(d_cost(nodeValues[numOfLayers - 1], expectedValues), order="C")

    for x in range(0, numOfLayers - 1):
        g_node *= d_stigmoid(r_stigmoid(nodeValues[-1 * (x + 1)]))
        g_biases.insert(0, np.array(g_node, order="C"))
        g_weights.insert(0,
                         np.transpose(np.resize(g_node, (numOfNodes[-1 * (x + 2)], numOfNodes[-1 * (x + 1)]))) *
                         np.resize(nodeValues[-1 * (x + 1)], (numOfNodes[-1 * (x + 1)], numOfNodes[-1 * (x + 2)]))
                         )
        g_node = np.matmul(np.transpose(weights[-1 * (x + 1)]), g_node)

    return (g_weights, g_biases)

@jit()
def applyGradient(weights_gradient, bias_gradient):
    for x in range(0, numOfLayers - 1):
        weights[x] +=  weights_gradient[x] * learningrate
        biases[x] += bias_gradient[x] * learningrate

def readImages():
    f_images = open("Images/images", "rb")
    f_lables = open("Images/labels", "rb")

    f_images.seek(16, 0)
    f_lables.seek(8, 0)

    images = []
    lables = []

    # parsing images
    buffer = struct.unpack("B" * (28 * 28 * 60000), f_images.read())
    for x in range(0, 60000):
        images.append(buffer[x * 28 * 28: x * 28 * 28 + 28 * 28])
    f_images.close()

    # parsing lables
    buffer = struct.unpack("b" * (60000), f_lables.read())
    for x in range(0, 60000):
        lables.append(buffer[x])
    f_lables.close()

    return (images, lables)

def run():
    logging = True
    images, lables = readImages()
    for w in range(numOfIterations):
        for x in range(numOfEvolutions):
            g_weights, g_bias, expected = [], [], [0 for x in range(10)]

            for y in range(0, runsPerEvolution):
                print(str(w) + "\tEvolution: " + str (x) + " Run: " + str(y + 1))
                buffer = np.array(images[x * runsPerEvolution + y], order="C") * 1.0/256
                nodeActivation = runNetwork(buffer)

                expected = [0 for x in range(10)]
                expected[lables[x * runsPerEvolution + y]] = 1

                gradient = backprogation(nodeActivation, np.array(expected, order="C"))
                g_weights.append(gradient[0])
                g_bias.append(gradient[1])

                print("Cost for run: " + str(np.dot(np.full((numOfNodes[-1]), 1.0/numOfNodes[-1]), cost(nodeActivation[-1], np.array(expected, order="C")))))

                print("\t\tAnswer: " + str((nodeActivation[-1].tolist()).index(max(nodeActivation[-1].tolist()))))
                if (nodeActivation[-1].tolist()).index(max(nodeActivation[-1].tolist())) == lables[x * runsPerEvolution + y]:
                    print("Correct", end="")
                else:
                    print("No", end="")

                for z in nodeActivation[-1].tolist():
                    print(" " + str(z) + " ", end="")

            for y in range(runsPerEvolution):
                applyGradient(g_weights[y], g_bias[y])

        for x in range(numOfLayers - 1):
            np.save(networkName + ".w." + str(x), weights[x])
            np.save(networkName + ".b."  + str(x), biases[x])

run()