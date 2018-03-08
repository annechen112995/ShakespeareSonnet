from shakespeare_processing import *

BORDER = "==============================================================="


def load_file(filename):
    '''
    Given a filename, load the file and return list of sentences
    deliminated by '\n'
    '''
    raw_text = open(filename).read()
    return raw_text


def import_syllables():
    '''
    Returns a dictionary of words to syllables.

    Syllables: list of strings in the form of '1' or 'E1'
        where the number represent the number of syllables and 'E' represent
        that it is only that number of syllables if the word appears at end
        of line
    '''
    filename = 'data/Syllable_dictionary.txt'
    raw_text = load_file(filename)
    text_list = raw_text.strip().split('\n')
    syllables = {}

    for line in text_list:
        sep = line.split(' ')
        word = sep[0]
        info = sep[1:]
        syllables[word] = info
    return syllables


def generate_rhyme_dict():
    '''
    Constructs a rhyme_to_word dictionary and a word_to_rhyme dictionary
    '''
    filename = 'data/shakespeare.txt'
    raw_text = open(filename).read()
    text_list = raw_text.strip().split('\n')

    # Preprocessing
    text_list = remove_int(text_list)
    text_list = remove_empty(text_list)
    text_list = lowercase(text_list)
    text_list = remove_punctuation(text_list)

    new_text = '\n'.join(text_list)

    # Separate into sonnets
    sonnets = separate_sonnets(new_text)

    # Initialize dictionary
    rhyme_to_word = {}
    word_to_rhyme = {}

    # Rhyming pattern of Shakespean Sonnet
    pattern_14 = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]
    pattern_12 = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]

    for i, sonnet in enumerate(sonnets):
        lines = sonnet.split('\n')
        print("Sonnet {}".format(i))
        print("Number of lines: {}".format(len(lines)))

        if len(lines) == 14:
            pattern = pattern_14
        else:
            pattern = pattern_12

        for (i, j) in pattern:
            print(BORDER)
            print(lines[i])
            print(lines[j])
            print(BORDER)
            last_word_i = lines[i].split(' ')[-1]
            last_word_j = lines[j].split(' ')[-1]
            rhyme_i = None
            rhyme_j = None

            # Check if words are in the dictionary
            if last_word_i in word_to_rhyme:
                rhyme_i = word_to_rhyme[last_word_i]
            if last_word_j in word_to_rhyme:
                rhyme_j = word_to_rhyme[last_word_j]

            # If both words have not been in dictionary, create new rhyme
            if rhyme_i is None and rhyme_j is None:
                keys = list(rhyme_to_word.keys())
                if keys == []:
                    rhyme = 0
                else:
                    rhyme = max(keys) + 1
                rhyme_to_word[rhyme] = [last_word_j, last_word_i]
                word_to_rhyme[last_word_i] = rhyme
                word_to_rhyme[last_word_j] = rhyme

            # if word j is in dictionary, add i to j's rhyme
            elif rhyme_i is None:
                word_to_rhyme[last_word_i] = rhyme_j
                rhyme_to_word[rhyme_j].append(last_word_i)

            # if word i is in dictionary, add j to i's rhyme
            elif rhyme_j is None:
                word_to_rhyme[last_word_j] = rhyme_i
                rhyme_to_word[rhyme_i].append(last_word_j)

            # if separate rhymes, combine
            elif rhyme_i != rhyme_j:
                list_words_i = rhyme_to_word[rhyme_i]
                list_words_j = rhyme_to_word[rhyme_j]
                combine = list_words_i + list_words_j
                rhyme_to_word[rhyme_i] = combine
                for word in list_words_j:
                    word_to_rhyme[word] = rhyme_i
                del rhyme_to_word[rhyme_j]

    return rhyme_to_word, word_to_rhyme


if __name__ == '__main__':
    (rtw, wtr) = generate_rhyme_dict()
