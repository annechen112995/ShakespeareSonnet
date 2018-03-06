import nltk
import string

from collections import Counter


def process_data(text):
    '''
    Takes in a text file and tokenizes it.

    Input: Text file, delimited by '\n'.

    Output:
    '''
    new_text = lowercase_no_punctuation(text)
    tokens = get_tokens(new_text)
    count = Counter(tokens)
    
    print(count.most_common(10))

    return new_text

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
