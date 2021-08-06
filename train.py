import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import ToxicDataset
from model import Tesseract
from EarlyStop import EarlyStopping
import nltk
import joblib
from config import config
from collections import Counter
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
        mean_  = torch.mean(lstm_out,1)
        max_ , _ = torch.max(lstm_out,1)
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
    df_train=df.loc[df.kfold!=0].reset_index(drop=True)
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

    def build_vocab(seq):
        words = Counter()  
        for i, sentence in enumerate(seq):
            for word in sentence:  
                words.update([word.lower()])  
        words = {k:v for k,v in words.items() if v>1}
        words = sorted(words, key=words.get, reverse=True)
    
        words = ['_PAD','_UNK'] + words
        word2idx = {o:i for i,o in enumerate(words)}
        idx2word = {i:o for i,o in enumerate(words)}
        return words, word2idx, idx2word

    def load_vectors():   
        path_to_glove_file = os.path.join(
            'D:\Dataset\embeddings', "glove.6B.100d.txt"
        )
        
        embeddings_index = {}
        with open(path_to_glove_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        
        print("Found %s word vectors." % len(embeddings_index))
        return embeddings_index
          
    def create_embedding_matrix(word_index, embedding_dict, embedding_dim=100):
        hits = 0
        misses = 0
        
        # Prepare embedding matrix
        embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))
        return embedding_matrix

    df_train['clean_text']= df_train['comment_text'].str.replace('\d+', '0')
    df_train['clean_text']= df_train['clean_text'].str.replace('\W+', ' ')
    df_train['clean_text']= df_train['clean_text'].apply(lambda x: tokenization(x))
    df_train['clean_text']= df_train['clean_text'].apply(lambda x:remove_stopwords(x))

    df_valid['clean_text']= df_valid['comment_text'].str.replace('\d+', '0')
    df_valid['clean_text']= df_valid['clean_text'].str.replace('\W+', ' ')
    df_valid['clean_text']= df_valid['clean_text'].apply(lambda x: tokenization(x))
    df_valid['clean_text']= df_valid['clean_text'].apply(lambda x:remove_stopwords(x))

    train_sequenceses = list(df_train['clean_text'])
    words, word2idx, idx2word = build_vocab(train_sequenceses)
    joblib.dump(word2idx, 'word2idx.pkl', compress=1)

    train_sequenceses = text_to_sequences(word2idx, list(df_train['clean_text']))
    val_sequenceses = text_to_sequences(word2idx, list(df_valid['clean_text']))


    train_sentences = pad_sequences(train_sequenceses, config.SEQ_LEN)
    val_sentences = pad_sequences(val_sequenceses, config.SEQ_LEN)
    train_labels = np.array(df_train['toxic'].values)
    val_labels = np.array(df_valid['toxic'].values)

    print("Loading embeddings")
    embedding_dict = load_vectors()
    embedding_matrix = create_embedding_matrix(
        word2idx, embedding_dict
        )
    joblib.dump(embedding_matrix, 'embedding_matrix.pkl', compress=1)

    train_dataset = ToxicDataset(
        train_sentences,
        train_labels

        )
    
    valid_dataset = ToxicDataset(
         val_sentences,
         val_labels
        )

    es = EarlyStopping(monitor='valid_loss', model_path=f'model.bin', patience=5, mode="min", delta=0.001)
    modl = SentimentNet(config.OUTPUT_SIZE, embedding_matrix, config.HIDDEN_DIM, config.N_LAYERS)
    modl.fit(train_dataset, valid_dataset, train_bs=16, valid_bs=16, epochs=10, callback=[es], fp16=True, device='cuda', workers=4)