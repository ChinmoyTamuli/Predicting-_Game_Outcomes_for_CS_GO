#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


from sklearn .linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# In[24]:


data=pd.read_csv('csgo_round_snapshots.csv')


# In[25]:


data


# In[26]:


np.sum(np.sum(data.isnull()))


# In[27]:


#data.drop(data.index[122410].axis=0,inplace=True)


# In[28]:


data.drop(data.select_dtypes(np.number),axis=1)


# In[29]:


data['bomb_planted']=data['bomb_planted'].astype(np.int16)


# In[30]:


data


# In[31]:


encoder=LabelEncoder()

data['map']=encoder.fit_transform(data['map'])
map_mappings={index: label for index, label in enumerate(encoder.classes_)}

map_mappings


# In[32]:


data['round_winner']=encoder.fit_transform(data['round_winner'])
round_winner_mappings={index: label for index, label in enumerate(encoder.classes_)}

round_winner_mappings


# In[33]:


y=data['round_winner']
X=data.drop('round_winner',axis=1)


# In[34]:


scaler=RobustScaler()
X=scaler.fit_transform(X)
pd.DataFrame(X)


# In[35]:


pca=PCA(n_components=84)
pca.fit(X)


# In[38]:


plt.figure(figsize=(10,10))
plt.hist(pca.explained_variance_ratio_,bins=84)
plt.show()


# In[39]:


def getKComponents(pca,alpha):
    total_variance =0
    
    for feature, variance in enumerate (pca.explained_variance_ratio_):
        
        total_variance +=variance
        if(total_variance >=1 -alpha):
            return feature +1
    return len(pca.explained_variance_ratio_)   


# In[40]:


K=getKComponents(pca, 0.05)


# In[41]:


X=pca.transform(X)[:, 0:K]
pd.DataFrame(X)


# In[42]:


X_train, X_test, y_train, y_test =train_test_split(X,y, train_size=0.8)


# In[43]:


log_model=LogisticRegression(verbose=True)
nn_model=MLPClassifier(verbose=True)


log_model.fit(X_train,y_train)
nn_model.fit(X_train,y_train)


# In[45]:


print(f"Logistic Model : {log_model.score(X_test,y_test)}")
print(f"Neural Net Model: {nn_model.score(X_test,y_test)}")


# In[ ]:




