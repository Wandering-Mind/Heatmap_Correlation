#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import mitosheet
mitosheet.sheet(analysis_to_replay="id-nkgzdxntsr")


# In[6]:


from mitosheet.public.v3 import *; register_analysis("id-nkgzdxntsr");
import pandas as pd

# Imported heart_disease.csv, heart_disease.csv
df = pd.read_csv(r'C:\\Users\\Rosed\\Downloads\\heart_disease.csv')



# In[7]:


# check structure
df.head()


# In[9]:


# check for missing values and data types 
df.info()
df.isnull().sum()


# In[11]:


# summarize key statistics 
df.describe()
df['target'].value_counts() # check class balance (1 = disease, 0 = no disease)


# In[13]:


# Visualizations
#compare disease risk across age gorups 
# correlation heatmap to analyze future relationships 
import seaborn as sns 
import matplotlib.pyplot as plt 
# age vs disease presence 
plt.figure(figsize=(8,5))
sns.histplot(df, x="age", hue="target", multiple="stack", bins=20)
plt.title("Disease Risk by Age")
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[17]:


# Data Preprocessing 
#handle missing values 
#scale numerical features 
#encode categorical variables

from sklearn.preprocessing import StandardScaler
df = df.dropna() # remove missing values 
scaler = StandardScaler()
df[['age', 'chol', 'trestbps']] = scaler.fit_transform(df[['age', 'chol', 'trestbps']])


# In[20]:


# Build Machine Learning Model 
#using random forest as a baseline model:

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report 

#split data
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Model 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Predictions 
y_pred = model.predict(X_test)

# evaluate 
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




