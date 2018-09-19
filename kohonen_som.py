# Author: Manohar Mukku
# Date: 19.09.2018
# Desc: Kohonen's Self-Organizing Map implementation
# GitHub: https://github.com/manoharmukku/kohonen-self-organizing-map

import getopt
import sys
import pandas

def parse_shape(shape):
    return [int(n) for n in shape.split(",")]

def usage():
    return

def main(argv):
    # Get the command line arguments
    try:
        opts, args = getopt.getopt(argv, "hdf:s:l:r:", ["help", "defaults", "file=", "shape=", "lr=", "rseed="])
    except getopt.GetoptError:
        sys.exit(2)

    # Defaults
    data_file = None # Required
    shape = None # Required
    lr = None # Required
    rseed = "42" # Optional
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
        elif (opt in ["-f", "--file"]):
            data_file = arg
            flag |= 1 # Set 1st bit from last to 1
        elif (opt in ["-s", "--shape"]):
            shape = arg
            flag |= 2 # Set 2nd bit from last to 1
        elif (opt in ["-l", "--lr"]):
            lr = arg
            flag |= 4 # Set 3rd bit from last to 1
        elif (opt in ["-r", "--rseed"]):
            rseed = arg

    # Sanity check the command line arguments

    # Check if all the required are specified. If not, whether the default flag is set
    if (defaults == False and flag != 7):
        sys.exit ("Oops! Please specify all the required arguments, or use --defaults flag, for using the default values of unspecified arguments")

    # If defaults value is set, assign defaults to unspecified arguments
    if (defaults == True):
        if (data_file == None):
            data_file = "data.csv"
        if (shape == None):
            shape = "100,100"
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

    # Sanity check and read the given data file into a dataframe
    df = None
    with open(data_file) as file:
        df = pd.readcsv(file)

if __name__ == "__main__":
    main(sys.argv[1:])
