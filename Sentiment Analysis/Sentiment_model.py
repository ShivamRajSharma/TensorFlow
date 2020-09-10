from tensorflow import keras
from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization,
    Dropout, Input, Embedding
    )

import tensorflow.keras.backend as k

def SentimentModel(
    vocab_size,
    embed_dims,
    hidden_dims,
    num_layers,
    dropout
    ):
    inp_ = Input(shape=(None,))
    x = Embedding(vocab_size, embed_dims)(inp_)
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    for _ in range(num_layers):
        x = LSTM(
            hidden_dims,
            dropout=dropout,
            return_sequences=True
        )(x)
    x = k.mean(x, axis=1)
    x = Dense(100, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inp_, x)
    return model
    
