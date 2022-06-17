# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:53:12 2022

@author: User
"""
import os
import numpy as np
import json
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



#%% #%% Model Deployment
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))

loaded_model.summary()

# to load tokenizer
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%%
input_review = 'The movie so good'

# preprcossing

input_review = re.sub('<.*?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()

from tensorflow.keras.preprocessing.text import tokenizer_from_json

tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review) 
input_review_encoded = pad_sequences(np.array(input_review_encoded).T,maxlen=180,
                              padding='post',
                              truncating='post')
    
outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

ONE_PATH = os.path.join(os.getcwd(),'one_path.pkl')
with open(ONE_PATH,'rb') as file:
    pickle_ohe = pickle.load(file)
    
print(pickle_ohe.inverse_transform(outcome))

