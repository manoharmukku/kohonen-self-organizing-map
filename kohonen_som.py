# Author: Manohar Mukku
# Date: 19.09.2018
# Desc: Kohonen's Self-Organizing Map implementation
# GitHub: https://github.com/manoharmukku/kohonen-self-organizing-map

import getopt
import sys
import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def is_within_radius(x1, y1, x2, y2, rad):
    if (dist(x1, y1, x2, y2) <= rad):
        return True
    return False

def initialize_weights(data, shape):
    # Initialize weight vectors from the input vectors
    rows = data.shape[0]
    n = shape[0] * shape[1]

    rand_rows = np.random.randint(rows, size=n)

    weights = data[rand_rows, :].T

    return weights

def kohonen_som(data, shape, lr, max_iter, rseed):
    m = data.shape[0]
    n = data.shape[1]

    rows = shape[0]
    cols = shape[1]

    # Initialize the weight matrix
    weights = initialize_weights(data, shape)

    # Initialize parameters
    t2 = 1000.0
    width = math.sqrt(shape[0]**2 + shape[1]**2) / 2.0
    t1 = 1000.0 / np.log(width)

    mapping_count = np.zeros([rows*cols, 1])

    # Iterate until convergence
    for t in range(max_iter):
        print ("\rIteration {}...".format(t))
        sys.stdout.flush()
        
        # Choose a random sample from the data
        x = data[np.random.randint(low=0, high=m, size=1), :].reshape(n, 1)

        # Calculate outputs from weights, x
        y = np.matmul(weights.T, x).T

        # Find the coordinates of the winning neuron
        winning = np.argmax(y)
        win_x = winning / rows
        win_y = winning % cols

        mapping_count[winning] += 1

        # Update the width for this iteration
        width = max(0.5, width * math.exp(-t / (1.0*t1)))

        # Calculate the neighborhood function
        h = np.zeros([rows*cols, 1])
        for j in range(rows*cols):
            j_x = j / rows
            j_y = j % cols
            if (is_within_radius(j_x, j_y, win_x, win_y, width)):
                h[j] = math.exp(-(dist(j_x, j_y, win_x, win_y)**2) / (2.0 * width**2))

        # Update the learning rate for this iteration
        lr = max(0.01, lr * math.exp(-t / (1.0 * t2)))

        # Update the weights matrix
        weights = weights + (lr * (x - weights) * (h.T))

    # Return the final weights and mapping counts
    return weights, mapping_count

def parse_shape(shape):
    return [int(n) for n in shape.split(",")]

def usage():
    return

def main(argv):
    # Get the command line arguments
    try:
        opts, args = getopt.getopt(argv, "hdf:s:l:r:i:", ["help", "defaults", "file=", "shape=", "lr=", "rseed=", "iterations="])
    except getopt.GetoptError:
        sys.exit(2)

    # Defaults
    data_file = None # Required
    shape = None # Required
    lr = None # Required
    rseed = "42" # Optional
    max_iter = None
    defaults = False
    flag = 0

    # Parse the command line arguments
    for opt, arg in opts:
        if (opt in ["-h", "--help"]):
            usage()
            sys.exit()
        elif (opt in ["-d", "--defaults"]):
            defaults = True
            print ("Using default values for unspecified arguments")
        elif (opt in ["-s", "--shape"]):
            shape = arg
            flag |= 1 # Set 1st bit from last to 1
        elif (opt in ["-l", "--lr"]):
            lr = arg
            flag |= 2 # Set 2nd bit from last to 1
        # elif (opt in ["-f", "--file"]):
        #     data_file = arg
        #     flag |= 4 # Set 3rd bit from last to 1
        elif (opt in ["-r", "--rseed"]):
            rseed = arg
        elif (opt in ["-i", "--iterations"]):
            max_iter = arg

    # Sanity check the command line arguments

    # Check if all the required are specified. If not, whether the default flag is set
    if (defaults == False and flag != 3):
        sys.exit ("Oops! Please specify all the required arguments, or use --defaults flag, for using the default values of unspecified arguments")

    # If defaults value is set, assign defaults to unspecified arguments
    if (defaults == True):
        if (data_file == None):
            data_file = "data.csv"
        if (shape == None):
            shape = "10,10"
        if (lr == None):
            lr = "0.1"

    # Sanity check and extract learning rate value
    try:
        lr = float(lr)
    except ValueError:
        sys.exit ("Oops! The learning rate should be a numeric value")

    # Sanity check and extract shape
    try:
        shape = parse_shape(shape)
    except ValueError:
        sys.exit ("Oops! The shape should be integer values, comma-separated without space")
    if (shape[0] <= 0 or shape[1] <= 0):
        sys.exit ("Oops! The shape should be positive integer values")

    # Sanity check and extract random seed value
    try:
        rseed = int(rseed)
    except ValueError:
        sys.exit ("Oops! Random seed should be an integer value")

    # Sanity check and extract max_iterations value
    try:
        if (max_iter == None):
            max_iter = int(1000 + 500 * shape[0] * shape[1])
        else:
            max_iter = int(max_iter)
    except ValueError:
        sys.exit ("Oops! Maximum iterations value should be an integer")
    if (max_iter <= 0):
        sys.exit ("Oops! Maximum iterations value should be positive")

    # Sanity check and read the given data file into a dataframe
    # df = None
    # with open(data_file) as file:
    #     df = pd.readcsv(file)

    # # Convert the dataframe to numpy ndarray
    # data = df.values()


    # Load MNIST dataset from sklearn
    digits = load_digits()
    data = digits.data

    print ("Training...")
    # Perform kohonen iterations until convergence and find the final weight matrix
    weights, mapping_count = kohonen_som(data, shape, lr, max_iter, rseed)

    print ("Plotting the heatmap...")
    # Print the heatmap of the mapping counts
    plt.imshow(mapping_count.reshape((shape[0], shape[1])), cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
