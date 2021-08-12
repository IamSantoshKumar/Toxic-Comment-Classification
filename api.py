from config import config
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import ToxicDataset
from model import Tesseract
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import uvicorn
from  fastapi import FastAPI
from pydantic import BaseModel
import warnings

warnings.filterwarnings('ignore')

app = FastAPI()

class sentence(BaseModel):
    sent : str


stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

porter_stemmer = PorterStemmer()
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

def tokenization(text):
    tokens = text.split(' ')
    return tokens

def text_to_sequences(word2idx, seq):
    for i, sentence in enumerate(seq):
        seq[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
    return seq

def pad_sequences(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features
         
class SentimentNet(Tesseract):
     def __init__(self, output_size, embedding_matrix, hidden_dim, n_layers, drop_prob=0.5):
         super(SentimentNet, self).__init__()
         self.output_size = output_size
         self.n_layers = n_layers
         self.hidden_dim = hidden_dim
        
         num_words = embedding_matrix.shape[0]
         embed_dim =  embedding_matrix.shape[1]
        
         self.embedding = nn.Embedding(num_words, embed_dim)
        
         self.embedding.weight = nn.Parameter(
         torch.tensor(
         embedding_matrix,
         dtype=torch.float32
         )
         )

         self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=drop_prob, bidirectional=True, batch_first=True)
         self.dropout = nn.Dropout(0.5)
         self.fc = nn.Linear(256*2, output_size)
         self.sigmoid = nn.Sigmoid()


     def init_hidden(self, batch_size):
         weight = next(self.parameters()).data
         hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().to('cuda'),
                      weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().to('cuda'))
         return hidden
         
     def forward(self, review, target=None):
         batch_size = review.size(0)
         h = self.init_hidden(batch_size)
         h = tuple([e.data for e in h])
         x = review.long()
         embeds = self.embedding(x)
         lstm_out, hidden = self.lstm(embeds, h)
        
         mean_ = torch.mean(lstm_out,1)
         max_, _ = torch.max(lstm_out,1)
         out = torch.cat((mean_, max_), 1)
         out = torch.sigmoid(self.fc(out))

         return out, None, None
     

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def fetch_predictions(text : sentence):

    data = {'comment_text': text.sent, 'toxic':-1}

    df_valid = pd.DataFrame(data, index=[0])

    embedding_matrix = joblib.load('embedding_matrix.pkl')
    word2idx = joblib.load('word2idx.pkl')

    df_valid['clean_text']= df_valid['comment_text'].str.replace('\d+', '0')
    df_valid['clean_text']= df_valid['clean_text'].str.replace('\W+', ' ')
    df_valid['clean_text']= df_valid['clean_text'].apply(lambda x: tokenization(x))
    df_valid['clean_text']= df_valid['clean_text'].apply(lambda x:remove_stopwords(x))

    val_sequenceses = text_to_sequences(word2idx, list(df_valid['clean_text']))
    val_sentences = pad_sequences(val_sequenceses, config.SEQ_LEN)
    val_labels = np.array(df_valid['toxic'].values)

    valid_dataset = ToxicDataset(
        val_sentences,
        val_labels
        )

    model = SentimentNet(config.OUTPUT_SIZE, embedding_matrix, config.HIDDEN_DIM, config.N_LAYERS)
    model.load_state_dict(torch.load('model.bin'))
    predicted = float(model.predict(valid_dataset, batch_size=1, device='cuda')[0][0])
    #preds = float(predicted.squeeze())
    return {"toxic": predicted, "not toxic": 1-predicted, "sentence": text}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)