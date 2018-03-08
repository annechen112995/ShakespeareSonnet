import numpy as np
import sys
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.layers import LSTM
from shakespeare_processing import *

BORDER = "==============================================================="
USAGE = (
    """Usage: python -W ignore RNN.py <training textfile>
    <generated textfile path> <optional verbose>""")


def train_LSTM(X, y, verbose=0):
    '''
    Takes training data X and Y and returns the fitted LSTM model

    Input:
        X : a list of sequences of int
        Y : one-hot encoding of the int coming after the sequence
    '''

    print("Building Model...")

    # Take a subset of sequences
#     X = X[0::5]
#     y = y[0::5]

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(180, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1], activation='softmax'))

    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    # fit the model
    model.fit(X, y, epochs=10, batch_size=512, verbose=verbose)
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, dataStart, int_to_char, char_to_int,
                  diversity=1.0, seed=0, verbose=0):
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

    print("Generating text...")

    n_vocab = len(int_to_char)
    size = len(dataStart[0])

    if seed == 0:
        # pick a random seed
        start = np.random.randint(0, len(dataStart) - 1)
        pattern = dataStart[start]

    else:
        seed = 'shall i compare thee to a summer\'s day?\n'
        pattern = [char_to_int[char] for char in seed]

    seq = [int_to_char[value] for value in pattern]

    if verbose == 1:
        print("Seed: ", ''.join(seq))

    # generate characters
    num_lines = 1
    max_num_lines = 14
    while num_lines < max_num_lines:
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

        if result == '\n':
            num_lines += 1

        if verbose == 1:
            print(BORDER)
            print("character ", i)
            print("selected char: ", result)
            print("new pattern: ",
                  ''.join([int_to_char[value] for value in pattern]))
            print(BORDER)

    # Return seq as string
    return ''.join(seq)


def save_textfile(filename, text):
    '''
    Given filename and text, save text in file

    Input: filename and text as string
    '''
    print("Saving generated text...")
    f = open(filename, 'w')
    f.write(text)
    f.close()
    return 0


def RNN(filename, save_filename, seed=0, diversity=0.5, verbose=0):
    '''
    Given filename for training data, predict and generate new text and save
    in save_filename
    '''
    text_list = load_data(file)
    (X, y, dataX, dataY, dataStart, int_to_char, char_to_int) = (
        process_data_RNN(text_list, verbose=verbose))
    model = train_LSTM(X, y, verbose=verbose)
    generated = generate_text(
        model, dataStart, int_to_char, char_to_int,
        diversity=diversity, seed=seed, verbose=verbose)
    return generated


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(USAGE)
        quit()
    file = sys.argv[1]
    save = sys.argv[2]
    verbose = 0
    seed = 0
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    if len(sys.argv) > 4:
        diversity = int(sys.argv[4])
    if len(sys.argv) > 5:
        diversity = int(sys.argv[5])

    # Disable warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    generated = RNN(file, save, seed=seed, diversity=diversity,
                    verbose=verbose)
    print(generated)
