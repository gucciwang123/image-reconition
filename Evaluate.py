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
    #function logic goes here

#AI scrip goes here
def run():
    #function logic goes here