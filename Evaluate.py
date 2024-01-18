import numpy as np
import PIL
import math
import struct

numOfPixles = 784 #change this value to for different number of pixles
numOfNodes = [numOfPixles, 16, 16, 10] #length must match number of layers
numOfLayers = 4 #change this for different number of layers in neural network
networkName = "train"

weights = [np.load(networkName + ".w." + str(x) + ".npy", allow_pickle=True) for x in range(numOfLayers - 1)]
biases = [np.load(networkName + ".b." + str(x) + ".npy", allow_pickle=True) for x in range(numOfLayers - 1)]

def stigmoid(x):
    return 1/(1 + math.exp(-1 * x))
vector_stigmoid = np.vectorize(stigmoid)

def runNeuralNetwork(input): #give pixle values in 32-bit floats. Length must mach numOfPixles
    nodeValues = [np.array(input)]
    for x in range(0, numOfLayers - 1):
        nodeValues.append(vector_stigmoid(np.matmul(weights[x], nodeValues[x]) + biases[x]))
    return nodeValues

#data parsing code
def data():
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

#AI scrip goes here
def run():
    #function logic goes here
    images, lables = data()
    buffer = np.array(images[x * runsPerEvolution + y], np.float) * 1.0 / 256
    nodeActivation = runNeuralNetwork(buffer)
    expected[lables[x * runsPerEvolution + y]] = 1

    print("Cost for run: " + str(
        np.dot(np.full((numOfNodes[-1]), 1.0 / numOfNodes[-1]), vector_cost(nodeActivation[-1], np.array(expected)))))

    print("\t\tAnswer: " + str((nodeActivation[-1].tolist()).index(max(nodeActivation[-1].tolist()))))
    if (nodeActivation[-1].tolist()).index(max(nodeActivation[-1].tolist())) == lables[x * runsPerEvolution + y]:
        print("Correct")
    else:
        print("No")