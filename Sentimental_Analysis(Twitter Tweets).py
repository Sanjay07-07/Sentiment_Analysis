#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required Datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  


# In[2]:


data = pd.read_csv("D:\Work\Imarticus\Machine Learning\Exam 2\\Sentiment.csv")


# In[3]:


#Previewing the Dataset
data.head()


# In[4]:


data.shape #We have 13,871 observations along with 21 columns


# In[5]:


#We need only 'text' and 'sentiments' columns for this dataset.
new_data = data[['text','sentiment']]
print(new_data)


# In[6]:


#We are checking the total number of 'positive' , 'negative' sentiments we have.
new_data['sentiment'].value_counts()


# In[7]:


#We need to classify tweets as either negative or positive, so we will filter out rows with neutral sentiment.

new_data = new_data[new_data['sentiment'] != 'Neutral']


# In[8]:


new_data.head()


# In[9]:


#convert sentiment to numeric
sentiment_label = new_data.sentiment.factorize()

sentiment_label


# In[10]:


pos = new_data[new_data['sentiment'] == 'Positive']
pos = pos['text']
neg = new_data[new_data['sentiment'] == 'Negative']
neg = neg['text']


# In[11]:


#We are now building a LSTM model to predict the sentiments.

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tweet = new_data.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


# We get actual texts from the dataframe.
# We are limiting the number of words to 5000.

# In[12]:


#index is assigned to all the words present in the dataframe.
print(tokenizer.word_index)


# In[13]:


print(tweet[1])
print(encoded_docs[1])


# In[14]:


print(padded_sequence[1])


# In[15]:


#Building the Model with LSTM

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length,     
                                     input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', 
                           metrics=['accuracy'])
print(model.summary())


# In[16]:


#Train the model

history = model.fit(padded_sequence,sentiment_label[0],
                  validation_split=0.2, epochs=50, batch_size=32)


# We have an accuracy of 97% for our model

# In[ ]:


#Based on the model we have built we are going to check the sentiment analysis of few sentences.


# In[17]:


#Statement 1: 'He is a great leader.'

test_word1 ="He is a great leader"
tw = tokenizer.texts_to_sequences([test_word1])
tw = pad_sequences(tw,maxlen=200)
prediction = int(model.predict(tw).round().item())
sentiment_label[1][prediction]


# In[24]:


#Statement 2: 'He is a terrible politician.'

test_word2 ="He is a terrible politician"
tw = tokenizer.texts_to_sequences([test_word2])
tw = pad_sequences(tw,maxlen=200)
prediction = int(model.predict(tw).round().item())
sentiment_label[1][prediction]


# In[28]:


#Statement 3: 'I hate your taste.'

test_word4 ="I hate you taste"
tw = tokenizer.texts_to_sequences([test_word4])
tw = pad_sequences(tw,maxlen=200)
prediction = int(model.predict(tw).round().item())
sentiment_label[1][prediction]


# We have successfully built a model which can predict the sentiment of a statement with an accuracy of 97%. Now we could have further made it more effective by using the Neutral Sentiment as well but for this we have only worked with 'Positive' and 'Negative' sentiments.

# In[ ]:




