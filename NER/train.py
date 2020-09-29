import warnings
warnings.filterwarnings('ignore')

import CONFIG
import NERModel
import dataloader

import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def run():
    df = pd.read_csv(CONFIG.INPUT_PATH, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
    pos_lb = LabelEncoder()
    pos_lb = pos_lb.fit(df['POS'].values)
    df['POS'] = pos_lb.transform(df['POS'].values)
    sentence = df.groupby('Sentence #')['Word'].apply(list).values
    pos = df.groupby('Sentence #')["POS"].apply(list).values
    x_val = sentence[:int(0.1*len(sentence))]
    y_val = pos[:int(0.1*len(sentence))]
    

    print('--------- [INFO] TOKENIZING --------')
    train_loader = dataloader.DataLoader(sentence, pos, CONFIG.Batch_size)

    pickle.dump(pos_lb, open('input/pos_lb.pickle', 'wb'))
    pickle.dump(train_loader.vocab.word_to_idx, open('input/word_to_idx.pickle', 'wb'))
    
    x_val = train_loader.vocab.numericalize(x_val)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, padding='post', value=train_loader.vocab.word_to_idx["<PAD>"])
    y_val = keras.preprocessing.sequence.pad_sequences(y_val, padding='post')

    vocab_size = len(train_loader.vocab.word_to_idx)
    classes = len(list(pos_lb.classes_))

    model = NERModel.NERModel( 
        vocab_size=vocab_size, 
        num_classes=classes
    )

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['sparse_categorical_accuracy']
    )
    
    print(f'------- [INFO] STARTING TRAINING -------')
    
    model.fit(train_loader, epochs=CONFIG.Epochs, batch_size=CONFIG.Batch_size, validation_data=(x_val, y_val))
    model.save(CONFIG.MODEL_PATH)


if __name__ == "__main__":
    run()
