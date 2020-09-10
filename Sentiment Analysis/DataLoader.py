import numpy as np
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Vocab:
    def __init__(self, thresh, tokenizer):
        self.word_to_idx = {}
        self.freq_count = {}
        self.thresh = thresh
        self.tokenizer = tokenizer
        self.stopwords_ = stopwords.words('english')

    def vocab_gen(self,  all_reviews):
        idx = 1
        for sentence in tqdm(all_reviews, total=len(all_reviews)):
            for word in self.tokenizer(sentence):
                word = str(word.text.lower())
                if word not in self.stopwords_:
                    if self.freq_count.get(word):
                        if self.thresh <= self.freq_count[word]:
                            if self.word_to_idx.get(word) is None:
                                self.word_to_idx[word] = idx
                                idx += 1
                        else:
                            self.freq_count[word] += 1
                    else:
                        self.freq_count[word] = 1 
    

    def numericalize(self, batch_data):
        final_data = []
        for sentence in batch_data:
            sentence_idx = []
            for word in self.tokenizer(sentence):
                if self.word_to_idx.get(str(word.text.lower())):
                    sentence_idx.append(self.word_to_idx[str(word.text.lower())])
                else:
                    sentence_idx.append(0)
            final_data.append(sentence_idx)
        #DYNAMIC PADDING WRT BATCH
        padded_bacth_data = pad_sequences(final_data, padding='post')
        return padded_bacth_data


        
class DataLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        x,
        y,
        batch_size=32,
        labels=True,
        shuffle=True
    ):
        self.x = x
        self.y = y
        self.indexes = np.arange(len(self.x))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = spacy.load("en_core_web_sm")
        self.vocab = Vocab(5, self.tokenizer)
        self.vocab.vocab_gen(x)
        self.on_epoch_end()
    
    def __len__(self):
        x = len(self.indexes)//self.batch_size
        x += int((len(self.indexes)%self.batch_size)!=0)
        return x

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
        x = self.x[indexes]
        y = self.y[indexes]
        x = self.vocab.numericalize(x)
        return x, y
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)