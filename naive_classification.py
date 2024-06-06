import random
import pandas as pd
import numpy as np
import os
import librosa
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore')

CFG = {
    'SR': 16000,
    'N_MFCC': 32, # Melspectrogram 벡터 개수
    'SEED': 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정

train_df = pd.read_csv('input/train.csv')
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
test_df = pd.read_csv('input/test.csv')

def get_mfcc_feature(df):
    features = []
    for path in tqdm(df['path']):
        # wav file load
        y, sr = librosa.load(path, sr=CFG['SR'])
        # mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])
        y_feature = []
        # feature 추출
        for e in mfcc:
            y_feature.append(np.mean(e))
        features.append(y_feature)

    mfcc_df = pd.DataFrame(features, columns=['mfcc_'+str(x) for x in range(1,CFG['N_MFCC']+1)])
    return mfcc_df


train_x = get_mfcc_feature(train_df)    # x : .wav file
valid_x = get_mfcc_feature(valid_df)
test_x = get_mfcc_feature(test_df)

train_y = train_df['label']     # y : label

X = torch.tensor(train_x.toarray(), dtype=torch.float32)
y = torch.tensor(train_y, dtype=torch.int64)


class NaiveBayesClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(NaiveBayesClassifier, self).__init__()
        self.num_classes = num_classes
        self.class_log_prior_ = nn.Parameter(torch.zeros(num_classes))
        self.feature_log_prob_ = nn.Parameter(torch.zeros(num_classes, num_features))

    def forward(self, x):
        return F.log_softmax(x @ self.feature_log_prob_.t() + self.class_log_prior_, dim=1)

    def fit(self, X, y):
        class_count = torch.bincount(y)
        self.class_log_prior_.data = torch.log(class_count.float() / y.size(0))

        feature_count = torch.zeros(self.num_classes, X.size(1))
        for c in range(self.num_classes):
            feature_count[c] = X[y == c].sum(dim=0)

        smoothed_fc = feature_count + 1
        smoothed_cc = class_count.view(-1, 1).float() + X.size(1)
        self.feature_log_prob_.data = torch.log(smoothed_fc / smoothed_cc)

    def predict(self, X):
        log_probs = self.forward(X)
        return torch.argmax(log_probs, dim=1)



num_features = X.size(1)
num_classes = len(set(train_y))     # num_classes=5
model = NaiveBayesClassifier(num_features, num_classes)
model.fit(X, y)


# 예측
X_test = torch.tensor(test_x.toarray(), dtype=torch.float32)
preds = model.predict(X_test)
print("Predictions:", preds)










