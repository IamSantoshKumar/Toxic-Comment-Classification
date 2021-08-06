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
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)
     
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
         
     def loss_fn(self, outputs, targets):
         loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets.squeeze())
         return loss
     
     def metrics_fn(self, outputs, targets):
         outputs = np.array(outputs.squeeze().detach().cpu()) >= 0.5
         targets = targets.squeeze().cpu().detach().numpy()
         acc = accuracy_score(targets, outputs)
         return {'accuracy_score': acc}
     
     def fetch_optimizer(self):
         opt = torch.optim.Adam(self.parameters(), lr=1e-4)
         return opt
     
     def fetch_scheduler(self):
         sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
             self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
         )
         return sch

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
         out = self.fc(out)
         
         loss=None
         
         if target is not None:
              loss = self.loss_fn(out, target)
              metrics = self.metrics_fn(out, target)
              return out, loss, metrics
         return out, None, None
     

if __name__=='__main__':

    seed_everything(seed=42)

    df = pd.read_csv(os.path.join('D:\Dataset\jigsaw','toxic_folds.csv'))
    df_valid=df.loc[df.kfold==0].reset_index(drop=True)

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
    predicted = model.predict(valid_dataset, batch_size=16, device='cuda')
