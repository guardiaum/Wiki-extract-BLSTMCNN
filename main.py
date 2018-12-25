#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from util import *
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, Flatten, concatenate
from keras.initializers import RandomUniform


def get_embeddings_from_glove(dictionary):
    fEmbeddings = open("embeddings/glove.6B/glove.6B.50d.txt", encoding="utf-8")

    wordEmbeddings = []
    word2Idx = {}
    for line in fEmbeddings:
        split = line.strip().split(" ")

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split) - 1)  # Zero vector for 'PADDING' word subtract 1 to remove index for word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

        # in case of embedding word is present on dictionary
        # adds its numbers to a np array (vector)
        if split[0].lower() in dictionary:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[split[0]] = len(word2Idx)

    return wordEmbeddings, word2Idx


#data = read_file()
data = read_file2()
# data = transform_to_other(data, 'state')
#data = get_data_from_class(data, 'state')

data_train, data_test = split_data_get_features(data)
print(len(data_train))
print(len(data_test))

sentences = add_char_information_in(data_train)
sentences_test = add_char_information_in(data_test)

# CREATE DICTIONARY OF LABELS AND TOKENS
label_set, dictionary = create_dictionaries(sentences)

# CREATE A MAPPING FOR LABELS
label2Idx = labels_mappings(label_set)

print(label2Idx)

# HARD CODED CASE LOOKUP
case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2,
            'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
            'contains_digit': 6, 'PADDING_TOKEN': 7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

# READ WORD EMBEDDINGS
wordEmbeddings, word2Idx = get_embeddings_from_glove(dictionary)

wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {'PADDING': 0, 'UNKNOWN': 1}
for c in " 0123456789abcdeëfghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_–()[]{}!?:;#'\"/\\%$`&=*+@^~|²":
    char2Idx[c] = len(char2Idx)

train_set = padding(create_matrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx))
test_set = padding(create_matrices(sentences_test, word2Idx, label2Idx, case2Idx, char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}

# batches
train_batch, train_batch_len = create_batches(train_set)
test_batch, test_batch_len = create_batches(test_set)


'''print(train_set[0][0])
word = ""
for i in train_set[0][0]:
    for char in word2Idx:
        word_ind = word2Idx[char]
        if i == word_ind:
            word += " " + char
print(word)

print(train_set[1][0])
word = ""
for i in train_set[1][0]:
    for char in word2Idx:
        word_ind = word2Idx[char]
        if i == word_ind:
            word += " " +  char
print(word)'''

# model

words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings], trainable=False)(words_input)
casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
casing = Embedding(input_dim=caseEmbeddings.shape[0], output_dim=caseEmbeddings.shape[1], weights=[caseEmbeddings], trainable=False)(casing_input)
char_input = Input(shape=(None, 52,), name='char_input')
embed_char_out = TimeDistributed(Embedding(len(char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_input)
dropout = Dropout(0.5)(embed_char_out)
conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(dropout)
maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words, casing, char])
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input, casing_input, char_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
model.summary()

epochs = 80
for epoch in range(epochs):
    print("Epoch %d/%d" % (epoch, epochs))
    a = Progbar(len(train_batch_len))
    for i, batch in enumerate(iterate_minibatches(train_batch, train_batch_len)):
        labels, tokens, casing, char = batch
        model.train_on_batch([tokens, casing, char], labels)
        a.update(i)
    print('\n')

#   Performance on test dataset
predLabels, correctLabels, sentences = tag_dataset(model, test_batch)

sents = []
for s in sentences:
    word = ""
    for number in s[0]:
        for char in word2Idx:
            word_ind = word2Idx[char]
            if number == word_ind:
                word += " " + char
    sents.append(word)


pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label, sents)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
