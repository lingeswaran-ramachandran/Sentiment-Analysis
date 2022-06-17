# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:17:23 2022

@author: User
"""

import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import  ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras.layers import Bidirectional,Embedding

CSV_URL = 'https://github.com/Ankit152/IMDB-sentiment-analysis/blob/master/IMDB-Dataset.csv?raw=true'
vocab_size = 10000
oov_token = 'OOV'    
max_len = 180

# EDA
# Step 1) Data loading

df = pd.read_csv(CSV_URL)
# df_copy = df_copy.copy() # backup

# Step 2) Data Inspection
df.head(10)
df.tail(10)
df.info()

df['sentiment'].unique()
df['review'][5]
df['sentiment'][5]
df.duplicated().sum() # to check for duplicated data
df[df.duplicated()]

# Step 3) Data Cleaning

# to remove duplicated data
df = df.drop_duplicates()
print(df)

review = df['review'].values # Features : X
sentiment = df['sentiment'].values # sentiment : y

for index,rev in enumerate(review):
    # remove html tags
    # ?dont be greedy
    # * zero or more occurences
    # .Any character except new line (/n)    
    review[index] = re.sub('<.*?>',' ',rev) 

    # convert into lower case
    # remove numbers 
    # ^ means NOT
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()

# Step4) Features selection
# Nothing to select

#%% 
# Step 5) Preprocessing
#         1) Convert into lower case
#         2) tokenization
   
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review) # learn all of the words
word_index = tokenizer.word_index
# print(word_index)
                    
train_sequences = tokenizer.texts_to_sequences(review) # to convert into numbers

#               3) Padding & truncating
length_of_review = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_review)) # to get the number of max length for padding

padded_review = pad_sequences(train_sequences,maxlen=max_len,padding='post',truncating='post')

#               4) One Hot Encoding for the Target

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

#               5) Train test split

X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 sentiment,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)
#%% Model development
# USE LSTM layers, dropout, dense, input
# achieve >90% f1 score
embedding_dim = 64

model = Sequential()
model.add(Input(shape=(180))) # np.shape(X_train)[1:]
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.summary()

plot_model(model) 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='acc')

hist = model.fit(X_train,y_train,
                 epochs=10,
                 batch_size=128,
                 validation_data=(X_test,y_test))

hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss') 
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation acc') 
plt.legend()
plt.show()

#%% model evaluation

y_true = y_test
y_pred = model.predict(X_test)

#%%
y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1)

#%%
print(classification_report(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))

#%% Model saving

import os
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

import json
token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)                              


ONE_PATH = os.path.join(os.getcwd(),'one_path.pkl')
with open(ONE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#%% Discussion/ Reporting

# Model achieved around 85% accuracy during training
# Recall and f1-score reportrs 86 and 86% respectively
# However the models 
# Earlystopping can be introduced in future to prevent overfitting
# Increase dropout rate to control overfitting 
# Trying with different DL architecture for example BERT model,transformer
# model, GPT3 model may help to imporove the model


#


