import CONFIG
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, Embedding,
    LayerNormalization
)

def NERModel(vocab_size, num_classes):
    inp_ = Input(shape=(None,))
    x =  Embedding(vocab_size, CONFIG.embed_dims)(inp_)
    x = Dropout(CONFIG.dropout)(x)
    for i in range(CONFIG.num_layers):
        x = GRU(CONFIG.hidden_dims, dropout=CONFIG.dropout, return_sequences=True)(x)
        tf.clip_by_value(x, -1, 1)
        x = LayerNormalization()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model =  Model(inp_, x)
    
    return model