import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from shakespeare_processing import *


def train_LSTM(X, y):
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
    model.add(LSTM(180, input_shape=(X.shape[1], X.shape[2],)))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # fit the model
    model.fit(X, y, epochs=25, batch_size=256)
    return model


def generate_text(model, dataX, int_to_char, n_vocab):
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
    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    seq = [int_to_char[value] for value in pattern]

    # generate characters
    for i in range(600):
        # Create and normalize x to be input of RNN
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)

        # Make prediction using trained model
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)

        # Convert prediction to character
        result = int_to_char[index]

        # Add prediction to pattern and set to size 40
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

        # Add result to seq
        seq.append(result)

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


def RNN(filename, save_filename):
    '''
    Given filename for training data, predict and generate new text and save
    in save_filename
    '''
    text_list = load_data(filename)
    (X, y, dataX, dataY, int_to_char, n_vocab) = process_data_RNN(text_list)
    model = train_LSTM(X, y)
    generated = generate_text(model, dataX, int_to_char, n_vocab)
    save_textfile(save_filename, generated)
    return generated


if __name__ == "__main__":
    generated = RNN('data/shakespeare.txt', 'generated/shakespeare.txt')
    print(generated)
