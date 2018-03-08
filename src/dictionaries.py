

def load_file(filename):
    '''
    Given a filename, load the file and return list of sentences
    deliminated by '\n'
    '''
    raw_text = open(filename).read()
    text_list = raw_text.strip().split('\n')
    return text_list


def import_syllables():
    '''
    Given a filename, returns a dictionary of words to syllables.

    Syllables: list of strings in the form of '1' or 'E1'
        where the number represent the number of syllables and 'E' represent
        that it is only that number of syllables if the word appears at end
        of line
    '''
    filename = 'data/Syllable_dictionary.txt'
    data = load_file(filename)
    syllables = {}

    for line in data:
        sep = line.split(' ')
        word = sep[0]
        info = sep[1:]
        syllables[word] = info
    return syllables
