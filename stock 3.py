#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas_datareader as web
import numpy as np
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


df = web.DataReader('AAPL', data_source='yahoo', start= '2012-01-01', end='2020-12-04')
df


# In[3]:


df.shape


# In[4]:


plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('DATE',fontsize = 18)
plt.ylabel('Close price',fontsize = 18)
plt.show()


# In[5]:


#creating a new dataframe with only the close column
data = df.filter(['Close'])

#Converting the dataframe to a numpy array
dataset = data.values

#get the number of rows to train the model
training_data_len = math.ceil( len(dataset) * .8)

training_data_len


# In[6]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[7]:


#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]

#Split the data into x_train and y_train data sets
x_train = []
y_train = [] 

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i <= 61:
    print(x_train)
    print(y_train)
    
#first array is x_train dataset and next array is y_train dataset  


# In[8]:


#convert the x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)


# In[9]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[10]:


#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM (50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[11]:


#compiling model
model.compile(optimizer= 'adam' , loss = 'mean_squared_error')


# In[12]:


model.fit(x_train, y_train, batch_size= 1 , epochs = 1)


# In[13]:


#create the testing data set
#create a new array containing scaled value from index 1738 to 2298
test_data = scaled_data[training_data_len - 60: , : ]
#create dataset x_test and y_test
x_test = []
y_test = dataset[training_data_len: , : ]
for i in range (60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


# In[14]:


#convert the data to numpy array
x_test = np.array(x_test)


# In[15]:


#reshape the data becuase our data is 2D but LSTM need 3D data to use
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))


# In[16]:


#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[17]:


#Get the root mean squared method (RMSE) 
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse


# In[18]:


#plot the dataset
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visulize the data
plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close price', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions',] , loc = 'lower right')
plt.show()


# In[ ]:




