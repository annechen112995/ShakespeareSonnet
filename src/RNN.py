import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from shakespeare_processing import *

BORDER = "==============================================================="
USAGE = "Usage: python RNN.py <training textfile> <generated textfile path>"


def train_LSTM(X, y, verbose=0):
    '''
    Takes training data X and Y and returns the fitted LSTM model

    Input:
        X : a list of sequences of int
        Y : one-hot encoding of the int coming after the sequence
    '''

    # Take a submit of sequences
    # X = X[0::10]
    # y = y[0::10]

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop', metrics=['accuracy'])

    # fit the model
    model.fit(X, y, epochs=50, batch_size=128, verbose=verbose)
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, dataX, int_to_char, verbose=0):
    '''
    Given model, dataX, int_to_char, n_vocab returns generated_text using
    predict function

    Input:
        model: the LSTM model that we trained
        dataX: list of sequences
        int_to_char: a dictionary matching interger to specific character
        n_vocab: number of unique characters we have

    Output: generate_text as string

    '''
    print(BORDER)
    print("Generating text")
    n_vocab = len(int_to_char)
    diversity = 0.2

    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    seq = [int_to_char[value] for value in pattern]
    size = len(pattern)

    if verbose == 1:
        print("Seed: ", ''.join(seq))

    # generate characters
    for i in range(600):
        # Create and normalize x to be input of RNN
        x = np.zeros((1, size, n_vocab))
        for t, char in enumerate(pattern):
            x[0, t, char] = 1

        # Make prediction using trained model
        prediction = model.predict(x, verbose=verbose)[0]
        index = sample(prediction, diversity)

        # Convert prediction to character
        result = int_to_char[index]

        # Add prediction to pattern and set to size 40
        pattern.append(index)
        pattern = pattern[1:1 + size]

        # Add result to seq
        seq.append(result)

        if verbose == 1:
            print("pred: ", prediction[0])
            print("selected index: ", index)
            print("selected char: ", result)
            print("new pattern: ",
                  ''.join([int_to_char[value] for value in pattern]))

    # Return seq as string
    return ''.join(seq)


def save_textfile(filename, text):
    '''
    Given filename and text, save text in file

    Input: filename and text as string
    '''
    f = open(filename, 'w')
    f.write(text)
    f.close()
    return 0


def RNN(filename, save_filename, verbose=0):
    '''
    Given filename for training data, predict and generate new text and save
    in save_filename
    '''
    text_list = load_data(file)
    (X, y, dataX, dataY, int_to_char, char_to_int) = (
        process_data_RNN(text_list, verbose=verbose))
    model = train_LSTM(X, y, verbose=verbose)
    generated = generate_text(model, dataX, int_to_char, verbose=verbose)
    save_textfile(save, generated)
    return generated


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(USAGE)
        quit()
    file = sys.argv[1]
    save = sys.argv[2]
    verbose = 0
    if len(sys.argv) > 3:
        verbose = sys.argv[3]
    generated = RNN(file, save, verbose=verbose)
    print(generated)
