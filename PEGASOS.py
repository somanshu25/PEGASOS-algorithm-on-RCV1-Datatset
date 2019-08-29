#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import fetch_rcv1 
import numpy as np
import random
from numpy import linalg

rcv1 = fetch_rcv1()

rcvLabel = rcv1.target.getcol(33)
rcvLabel = rcvLabel.astype(np.float)
rcvLabel[rcvLabel == 0] = -1

xData_train = rcv1.data[:100000,:]
yLabel_train = rcvLabel[:100000]

xData_test = rcv1.data[100001:, :]
yLabel_test = rcvLabel[100001:]





# In[ ]:


regPar = 0.0000010
w = np.zeros((rcv1.data.shape[1], 1), dtype = float)
B = 1000
i = 100


# In[ ]:



# print(w.shape)

for t in range(1, i):
    
    sample_indices = random.sample(range(0, xData_train.shape[0]-1), B)
    x_sample = xData_train[sample_indices]
    y_sample = yLabel_train[sample_indices]
    predictXY = x_sample.dot(w)
    false_indices_map = y_sample.multiply(predictXY) < 1
    false_indices  = np.where(false_indices_map.todense())[0]
    x_sample_falseclassified =  x_sample[false_indices]
    y_sample_falseclassified =  y_sample[false_indices]
    yx = np.sum(x_sample_falseclassified.multiply(y_sample_falseclassified), axis=0)
    gradient = np.dot(regPar, w) - yx/B 
    nue = 1.0/(regPar*i)

    
    #predictionsTrain = np.where((yLabel_train.multiply(xData_train.dot(w)) < 1).todense())[0]
    #print("Training Error", predictionsTrain.shape)
    print(gradient.shape)
    #test_predictions = np.where((y_test.multiply(x_test.dot(w.transpose())) < 1).todense())[0]
    #print("Test Error", test_predictions.shape)
    w1 = w - nue*(gradient)
    w = min(1, 1/((linalg.norm(w1))*np.sqrt(lambda_))) * w1



# In[ ]:


gradient.shape


# In[ ]:




