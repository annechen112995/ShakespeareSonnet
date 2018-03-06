import os, sys


def load_data(filename, skiprows = 1):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.

    Inputs:
        filename: given as a string.

    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    text = open(os.path.join(os.getcwd(), filename)).read()
    return text


def process_data(file):

    pass

if __name__ == '__main__':
    inputFile = sys.argv[1]
    initText = load_data(inputFile)
    processedText = process_data(initText)

    pass
