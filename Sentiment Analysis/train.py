import tensorflow as tf 
from tensorflow import keras
import DataLoader
import Sentiment_model
import pickle
import CONFIG
import pandas as pd 
import numpy as np
import spacy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_labels(labels):
    lb = LabelEncoder().fit(labels)
    labels = lb.transform(labels)
    return labels.reshape(-1, 1), lb

def run():
    df = pd.read_csv('input/IMDB Dataset.csv')
    data = df.review.values[:100]
    labels = df.sentiment.values[:100]
    labels, lb = preprocess_labels(labels)

    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.1)

    train_loader = DataLoader.DataLoader(x_train, y_train)
    x_val = train_loader.vocab.numericalize(x_val)
    pickle.dump(train_loader.vocab.word_to_idx, open('model/word_to_idx.pickle','wb'))
    pickle.dump(lb, open('model/LabelEncoder.pickle','wb'))
    vocab_size = len(train_loader.vocab.word_to_idx) +1


    model = Sentiment_model.SentimentModel(
                    vocab_size,
                    CONFIG.embed_dims,
                    CONFIG.hidden_dims,
                    CONFIG.num_layers,
                    CONFIG.dropout
                )
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_loader, batch_size=CONFIG.BATCH_SIZE, epochs=CONFIG.EPOCHS, validation_data = (x_val, y_val))
    model.save('model/sentiments.h5')


if __name__ == "__main__":
    run()