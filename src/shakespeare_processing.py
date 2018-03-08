import os
import nltk
import string

from collections import Counter
from nltk.corpus import stopwords

# Download required info libraries
# nltk.download('stopwords')
# nltk.download('punkt')

BORDER = "==================================================================="


def load_data(filename):
    '''
    Takes in a file path and creates a list of sentences deliminated by '\n'

    Input: Filename
    Output: List of sentences deliminated by '\n'
    '''

    text = open(os.path.join(os.getcwd(), filename)).read()
    new_text = text.strip().split('\n')

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


def remove_int(text):
    '''
    Remove numbers from strings
    '''
    new_text = []
    for line in text:
        # Remove integer values
        no_digits = ''.join([i for i in line if not i.isdigit()])

        # Remove punctuation
        new_text.append(no_digits)
    return new_text


def remove_int_new(text):
    '''
    Remove numbers from strings
    '''
    new_text = ''
    for line in text:
        # Remove integer values
        no_digits = ''.join([i for i in line if not i.isdigit()])

        # Remove punctuation
        new_text += no_digits
    return new_text


def remove_empty(text):
    '''
    Removes all empty string
    '''
    new_text = []
    for line in text:
        if not line.isspace():
            new_text.append(line)
    return new_text


def lowercase(text):
    '''
    Convert text to all lowercase
    '''
    new_text = []
    for line in text:
        # Make text lowercase
        line = line.lower()

        new_text.append(line)
    return new_text


def lowercase_new(text):
    '''
    Convert text to all lowercase
    '''
    new_text = ''
    for line in text:
        # Make text lowercase
        line = line.lower()

        new_text += line
    return new_text


def remove_punctuation(text):
    '''
    Remove punctuation
    '''
    new_text = []
    for line in text:
        sentence = ''.join([i for i in line])

        # Remove punctuation
        new_text.append(sentence.translate(
            str.maketrans('', '', string.punctuation)))
    return new_text


def separate_sonnets(text):
    return text.split('\n\n\n')
