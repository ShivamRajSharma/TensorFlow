import CONFIG

import tensorflow as tf 
import pandas as pd
import pickle
import numpy as np 
import spacy
from tensorflow.keras.models import load_model

def predict(sentence):
    model = load_model(CONFIG.MODEL_PATH)
    word_to_idx = pickle.load(open('input/word_to_idx.pickle', 'rb'))
    pos_lb = pickle.load(open('input/pos_lb.pickle', 'rb'))
    tokenizer = spacy.load('en_core_web_sm')
    sentence_idx = []
    sentence_tokenized = []
    for word in tokenizer(sentence):
        word = str(word.text.lower())
        if word in word_to_idx:
            sentence_tokenized.append(word)
            sentence_idx.append(word_to_idx[word])
    sentence_idx = np.array(sentence_idx)[None, :]
    predictions =  model.predict(sentence_idx)[0]
    predictions =  np.argmax(predictions, axis=-1)
    labels = pos_lb.inverse_transform(predictions)
    data = [sentence_tokenized, labels]
    df = pd.DataFrame(data).transpose()
    df.columns = ['Word', 'POS']
    print(f"Sentence -> {sentence} \n")
    print(df)

        

if __name__ == "__main__":
    print("\n")
    sentence = str(input("Enter a sentence : "))
    predict(sentence)
    print("\n")
