import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize
from nltk import pos_tag
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import Progbar


'''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
'''
def read_file(file):
    data = pd.read_csv(file, names=['sentence', 'entity', 'value', 'label'], )
    data['property'] = 'state'
    data = data[data['label'] == 't']
    subset = data[['property', 'value', 'sentence', 'property']]
    return split_data_get_features(subset)


'''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
'''
def read_validation_file(file, class_):
    data = pd.read_csv(file)
    data = data[data['property'] == class_]
    subset = data[['property', 'value', 'sentence']]

    data = subset[['sentence']].values
    values = subset[['value']].values
    labels = subset[['property']].values

    return [sentence2features(labels[index][0], values[index][0], row[0]) for index, row in enumerate(data)]


def get_data_from_class(data, class_):
    return data[data['property'] == class_]


def sentence2features(prop, value, sent):
    prop_tk = word_tokenize(prop.replace("_", " "))
    value_tk = word_tokenize(value)
    tokens = word_tokenize(sent)
    postags = pos_tag(tokens)
    return [[token, 'PROP' if token in prop_tk else 'VALUE' if token in value_tk else 'O'] for token, tag in postags]


def split_data_get_features(subset):

    data = subset[['sentence']].values
    values = subset[['value']].values
    labels = subset[['property']].values

    sss = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.25)
    train_index, test_index = next(sss.split(data, labels))
    data_train, data_test = data[train_index], data[test_index]
    values_train, values_test = values[train_index], values[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    train_features = [sentence2features(labels_train[index][0], values_train[index][0], row[0]) for index, row in enumerate(data_train)]
    test_features = [sentence2features(labels_test[index][0], values_test[index][0], row[0]) for index, row in enumerate(data_test)]

    return train_features, test_features


''' 
Gets characters information and adds to sentences
Returns a matrix where the row is the sentence and 
each column is composed by token setence, characters information from tokens and label for token
'''
def add_char_information_in(sentences):
    for i, sentence in enumerate(sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]  # data[0] is the token
            sentences[i][j] = [data[0], chars, data[1]]  # data[1] is the annotation (label)
    return sentences


def create_dictionaries(sentences):
    label_set = set()
    dictionary = {}
    for sentence in sentences:
        for token, char, label in sentence:
            label_set.add(label)
            dictionary[token.lower()] = True
    return label_set, dictionary


def labels_mappings(label_set):
    label2Idx = {}
    for label in label_set:
        label2Idx[label] = len(label2Idx)
    return label2Idx


def get_casing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():
        casing = 'allLower'
    elif word.isupper():
        casing = 'allUpper'
    elif word[0].isupper():
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


def create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1

            charIdx = []
            for x in char:
                if x not in char2Idx:
                    charIdx.append(char2Idx[' '])
                else:
                    charIdx.append(char2Idx[x])

            wordIndices.append(wordIdx)
            charIndices.append(charIdx)
            caseIndices.append(get_casing(word, case2Idx))
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])
    #print('\n'.join(map(str, dataset[0:5])))
    return dataset


def padding(sentences_matrix):
    maxlen = 52  # ????????
    for sentence in sentences_matrix:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(sentences_matrix):
        sentences_matrix[i][2] = pad_sequences(sentences_matrix[i][2], 52, padding='post')

    return sentences_matrix


def create_batches(data):
    l = []
    for i in data:
        l.append(len(i[0]))  # appends sentence size
    l = set(l)  # converts to set to remove duplicates
    batches = []
    batch_len = []
    z = 0
    for i in l:  # for each different sentence size
        for batch in data:  # for each sample
            if len(batch[0]) == i:  # if sentence size on batch is equal to the ones kept on 'l'
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        casing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            casing.append(c)
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels), np.asarray(tokens), np.asarray(casing), np.asarray(char)


def tag_dataset(model, dataset):
    correctLabels = []
    predLabels = []
    sentences = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing, char], verbose=False)[0]
        pred = pred.argmax(axis=-1)
        correctLabels.append(labels)
        sentences.append(tokens)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels, sentences