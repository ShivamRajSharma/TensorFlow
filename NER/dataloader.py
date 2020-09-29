from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import spacy

class Vocabulary:
    def __init__(self):
        self.word_to_idx = {"<UNK>": 0, "<PAD>": 1}
        self.idx_to_word = {0: "<UNK>", 1: "<PAD>"}
        self.tokenizer = spacy.load("en_core_web_sm")
    
    def vocab(self, sentences):
        num = 2
        for sentence in tqdm(sentences, total=len(sentences)):
            for word in sentence:
                word = word.lower()
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = num
                    self.idx_to_word[num] = word
                    num += 1
        print(len(self.word_to_idx))
    
    def numericalize(self, batch_sentence):
        final_data = []
        for sentence in batch_sentence:
            sentence_idx = []
            for word in sentence:
                word = word.lower()
                if word in self.word_to_idx:
                    sentence_idx.append(self.word_to_idx[word])
                else:
                    sentence_idx.append(self.word_to_idx['<UNK>'])
            final_data.append(sentence_idx)
        
        return final_data

class DataLoader(keras.utils.Sequence):
    def __init__(
        self, 
        sentences, 
        pos, 
        batch_size, 
        shuffle=True
    ):
        self.sentences = sentences
        self.pos = pos 
        self.indexex = np.arange(len(sentences))
        self.train_index = self.indexex[int(0.1*len(self.sentences)):]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vocab = Vocabulary()
        self.vocab.vocab(sentences)
        self.on_epoch_end()
    
    def __len__(self):
        x = len(self.train_index)//self.batch_size
        x += int((len(self.train_index)%self.batch_size)!=0)
        return x
    
    def __getitem__(self, idx):
        sentence_batch = self.sentences[self.train_index[self.batch_size*idx : self.batch_size*(idx+1)]]
        pos_batch =  self.pos[self.train_index[self.batch_size*idx: self.batch_size*(idx+1)]]
        sentence_batch = self.vocab.numericalize(sentence_batch)
        padded_sentence = pad_sequences(sentence_batch, padding='post', value=self.vocab.word_to_idx["<PAD>"])
        padded_pos = pad_sequences(pos_batch, padding='post')
        return padded_sentence, padded_pos
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexex)