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
        no_digits = ''.join([i for i in line if not i.isdigit()])
        new_text.append(no_digits.translate(str.maketrans('','',string.punctuation)))

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
    used words in the text. Also removes integers from the text (from verse
    headers, etc)
    '''
    tokens = get_tokens(text)
    new_text = [w for w in tokens if not w in stopwords.words('english')]

    # Check new token counts
    count = Counter(new_text)
    print(count.most_common(10))

    return new_text

def fixed_length_training_seq(text):
    '''
    Create fixed length training sequences of length 40 char from the sonnet
    corpus.

    Input: Text file in the form of list of words

    Output: List of 40 character sequences, or a list of one <40 char sequence
    '''
    text_string = ''
    seqs = []

    for word in text:
        text_string += word + ' '

    if len(text_string) < 40:
        seqs.append(text_string)
    else:
        start = 0
        while (start + 40) < len(text_string):

            seqs.append(text_string[start:start + 40])
            start += 1

    return seqs
