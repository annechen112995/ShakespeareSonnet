import nltk
import string

nltk.download('stopwords')

from collections import Counter
from nltk.corpus import stopwords


def process_data(text):
    '''
    Takes in a text file and tokenizes it.

    Input: Text file, delimited by '\n'.

    Output:
    '''
    new_text = lowercase_no_punctuation(text)
    filtered_text = remove_stopwords(new_text)

    return filtered_text

def lowercase_no_punctuation(text):
    '''
    Convert text to all lowercase and remove punctuation
    '''
    new_text = []
    for line in text:
        line = line.lower()
        new_text.append(line.translate(str.maketrans('','',string.punctuation)))

    return new_text

def get_tokens(text):
    '''
    Tokenize from text
    '''
    token_text = ''

    for line in text:
        token_text = token_text + line + ' '

    tokens = nltk.word_tokenize(token_text)

    return tokens

def remove_stopwords(text):
    '''
    Remove common stopwords from the tokens to get a better sense of the most
    used words in the text.
    '''
    tokens = get_tokens(text)
    new_text = [w for w in tokens if not w in stopwords.words('english')]

    # Check new token counts
    count = Counter(new_text)
    return new_text
