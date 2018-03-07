import os
import nltk
import string
import numpy as np
from keras.utils import np_utils

from collections import Counter
from nltk.corpus import stopwords

# Download required info libraries
nltk.download('stopwords')
nltk.download('punkt')


def load_data(filename):
    '''
    Takes in a file path and creates a list of sentences deliminated by '\n'

    Input: Filename
    Output: List of sentences deliminated by '\n'
    '''

    text = open(os.path.join(os.getcwd(), filename)).read()
    new_text = text.strip().split('\n')

    return new_text


def process_data(text):
    '''
    Takes in a text file as list of sentences and tokenizes it.

    Input: list of sentences, delimited by '\n'.

    Output: List of strings of words, delimited by ‘\n’
    Ex: [‘put’,\n ‘word’,\n  …,\n ‘here’,\n ‘potato’]
    '''
    new_text = lowercase_no_punctuation_no_int(text)
    filtered_text = remove_stopwords(new_text)

    return filtered_text


def lowercase_no_punctuation_no_int(text):
    '''
    Convert text to all lowercase and remove punctuation and numbers
    '''
    new_text = []
    for line in text:
        # Make text lowercase
        line = line.lower()

        # Remove integer values
        no_digits = ''.join([i for i in line if not i.isdigit()])

        # Remove punctuation
        new_text.append(no_digits.translate(
            str.maketrans('', '', string.punctuation)))
    return new_text


def get_tokens(text):
    '''
    Tokenize from text

    Input: list of sentences, delimited by '\n'

    '''
    token_text = ''

    for line in text:
        token_text = token_text + line + ' '

    tokens = nltk.word_tokenize(token_text)

    return tokens


def remove_stopwords(text):
    '''
    Remove common stopwords from the tokens to get a better sense of the most
    used words in the text. Also removes integers from the text (from verse
    headers, etc)
    '''
    tokens = get_tokens(text)
    new_text = [w for w in tokens if w not in stopwords.words('english')]

    # Check new token counts
    count = Counter(new_text)
    print(count.most_common(10))

    return new_text


def lowercase_no_int(text):
    '''
    Convert text to all lowercase and remove numbers
    '''
    new_text = []
    for line in text:
        # # Make text lowercase
        # line = line.lower()

        # Remove integer values
        no_digits = ''.join([i for i in line if not i.isdigit()])

        # Remove punctuation
        new_text.append(no_digits)
    return new_text


def process_data_RNN(text):
    '''
    Create fixed length training sequences of length 40 char from the sonnet
    corpus.

    Input: Text file in the form of list of words

    Output: X, Y, dataX, dataY, int_to_char, n_vocab
    '''
    border = "==============================================================="
    new_text_list = lowercase_no_int(text)
    new_text = '\n'.join(new_text_list)
    print(border)
    print("Processed text: ", new_text)
    print(border)

    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(new_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # summarize the loaded data
    n_chars = len(new_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 40
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = new_text[i:i + seq_length]
        seq_out = new_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))

    # normalize
    X = X / float(n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    return X, y, dataX, dataY, int_to_char, n_vocab
