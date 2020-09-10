import tensorflow as tf 
import pickle
import spacy
from tensorflow import keras 
import numpy as np 
import sys

def run(sentence):
    word_to_idx = pickle.load(open('model/word_to_idx.pickle', 'rb'))
    lb = pickle.load(open('model/LabelEncoder.pickle', 'rb'))
    model = keras.models.load_model('model/sentiments.h5')
    tokenizer = spacy.load("en_core_web_sm")
    print(f'1 -> {lb.inverse_transform([1])}')
    print(f'0 -> {lb.inverse_transform([0])}')
    sentence_idx = []
    for word in tokenizer(sentence):
        word = str(word.text.lower())
        if word_to_idx.get(word):
            sentence_idx.append(word_to_idx[word])
        else:
            sentence_idx.append(0)
    sentence_idx = np.array(sentence_idx)[None, :]
    predicted = (model.predict(sentence_idx)[0][0] > 0.5)*1
    sentiment = lb.inverse_transform([predicted])
    print(f'SENTENCE -> {sentence} | SENTIMENT-> {sentiment}')

if __name__ == "__main__":
    sentence = 'That was a great movie'
    run(sentence)

