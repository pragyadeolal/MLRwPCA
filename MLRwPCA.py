#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df= pd.read_csv('D:\Computer  Science\Project_2\insurance.csv')
print (df)


# In[3]:


print(df.info())


# In[4]:


df.head(5)


# In[5]:


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
print(x)


# In[6]:


df['sex'] = df['sex'].apply({'male':0, 'female':1}.get) 
df['smoker'] = df['smoker'].apply({'yes':1, 'no':0}.get)
df['region'] = df['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)
print(df)


# In[7]:


#Training and Test data split
x_tr,y_tr,x_te,y_te= df.iloc[:1000,:-1],df.iloc[:1000,-1],df.iloc[1000:,:-1],df.iloc[1000:,-1]
print(x_tr)


# In[8]:


sc = StandardScaler() 
x_tr = sc.fit_transform(x_tr)
x_te = sc.transform(x_te)


# In[9]:


pca = PCA(n_components = 2)
x_tr= pca.fit_transform(x_tr)
x_te = pca.transform(x_te)

explained_variance = pca.explained_variance_ratio_


# In[11]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_tr,y_tr)


# In[12]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
y_pred = model.predict(x_te)
print('y_pred=',y_pred)


# In[13]:


print('y_test',y_te)
mae = mean_absolute_error(y_te,y_pred)
mse = mean_squared_error(y_te,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_te,y_pred)
print("Mean Absolute Error", mae)
print("Mean Squared Error", mse)
print("Root Mean Square Error", rmse)
print("R squared",r2)


# In[14]:


plt.scatter(y_te,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Y Pred')


# In[16]:


import seaborn as sns
df_ = pd.DataFrame(pca.components_)
plt.figure(figsize =(14, 6))

sns.heatmap(df_)


# In[17]:


sns.histplot(y_te-y_pred)


# In[ ]:




