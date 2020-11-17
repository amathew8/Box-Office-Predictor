#!/usr/bin/env python
# coding: utf-8

# In[721]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.linear_model import Lasso 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from xgboost import XGBRegressor


# In[722]:


# Reading in data from excel
movieDF = pd.read_excel('movies_data.xlsx')
movieDF = movieDF.drop(movieDF.columns[0], axis=1)
movieDF = movieDF.drop(movieDF.columns[2], axis=1)
movieDF = movieDF.drop(movieDF.columns[2], axis=1)
movieDF = movieDF.drop(movieDF.columns[5], axis=1)
movieDF = movieDF.drop(movieDF.columns[7], axis=1)
movieDF = movieDF.rename(columns={"Genre(s)": "Genre", "Prod. Company(s)": "Production"})
movieDF.replace(to_replace =0, 
                 value =np.NaN,inplace=True) 
movieDF.replace(to_replace = 'N/A', 
                 value =np.NaN,inplace=True) 

movieDF = movieDF[np.isfinite(movieDF['Budget'])]
movieDF = movieDF.dropna()



# One hot encoding the genre values for regression #

genre_array = movieDF['Genre'].values
#genre_array = [value.split(',') for value in genre_array]
for i in range(len(genre_array)):
    genre_array[i] = genre_array[i].replace(" ", "")
genre_array = [value.split(',') for value in genre_array]
movieDF['Genre'] = genre_array
genreEncoding = MultiLabelBinarizer()
movieDF = movieDF.join(pd.DataFrame(genreEncoding.fit_transform(movieDF.pop('Genre')),
                          columns=genreEncoding.classes_,
                            index = movieDF.index))

# One hot encoding the production company values for regression #
production_array = movieDF['Production'].values
for i in range(len(production_array)):
    production_array[i] = production_array[i].replace("\"","")
    production_array[i] = production_array[i].replace("\'","")
    production_array[i] = production_array[i][1:-1]
production_array = [value.split(',') for value in production_array]
for i in range(len(production_array)):
    for j in range(len(production_array[i])):
        production_array[i][j] =  production_array[i][j].strip()
movieDF['Production'] = production_array
prodEncoding = MultiLabelBinarizer()
movieDF = movieDF.join(pd.DataFrame(prodEncoding.fit_transform(movieDF.pop('Production')),
                          columns=prodEncoding.classes_,
                            index = movieDF.index))

# One hot encoding the actors for regression #
actor_array = movieDF['Actors'].values
for i in range(len(actor_array)):
    if len(actor_array[i]) > 3:
        actor_array[i] = actor_array[i][1:-1]
        actor_array[i] = actor_array[i].replace("\"","")
        actor_array[i] = actor_array[i].replace("\'","")
actor_array = [row.split(',') if len(row) > 3 else [row+"_Actors"] for row in actor_array]
for i in range(len(actor_array)):
    for j in range(len(actor_array[i])):
        actor_array[i][j] =  actor_array[i][j].strip()
actorEncoding = MultiLabelBinarizer()
movieDF['Actors'] = actor_array
movieDF = movieDF.join(pd.DataFrame(actorEncoding.fit_transform(movieDF.pop('Actors')),
                          columns=actorEncoding.classes_,
                            index = movieDF.index))

# One hot encoding MPAA ratings #
mpaa_array = movieDF['MPAA'].values
for i in range(len(mpaa_array)):
    if '/' in mpaa_array[i]:
        mpaa_array[i] = mpaa_array[i]+"_MPAA"
movieDF['MPAA'] = mpaa_array
mpaaDummies = pd.get_dummies(movieDF['MPAA'],drop_first=True)
movieDF = movieDF.join(mpaaDummies)
movieDF = movieDF.drop('MPAA',axis=1)

# One Hot Encoding Director values #
directorDummies = pd.get_dummies(movieDF['Director'])
movieDF = movieDF.join(directorDummies,rsuffix='Director_')
movieDF = movieDF.drop('Director',axis=1)


# In[723]:


# Creating input vs output DataFrames #
X = movieDF.drop(columns=['Movie','Year','Box Office.1'])
Y = movieDF['Box Office.1']


# In[702]:


# Implementation of Select K Best Features #
reduced_features = SelectKBest(score_func = f_regression,k = 2500)
fit = reduced_features.fit(X,Y)
x_new = reduced_features.transform(X)
cols = reduced_features.get_support(indices=True)
selected_features_DF = X.iloc[:,cols]


# In[724]:


# Implementation of PCA #
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=3000)
principalComponents = pca.fit_transform(X)


# In[678]:


# Creating a table for most import features based on Uni-variate Selection #
dfscores = pd.DataFrame(fit.scores_)
dfcol = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcol,dfscores],axis=1)
featureScores.columns = ['Features','Score']
print(featureScores.nlargest(20,'Score'))


# In[725]:


# Splitting Training and Testing data #
X_train, X_test, y_train, y_test = train_test_split(principalComponents,Y.values,test_size=.25,random_state=42)


# In[726]:


# Final implelmation of XGBRegressor() #
regressor = XGBRegressor()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)


# In[727]:


# Storing r-squared metric #
score = r2_score(y_test,predictions)


# In[728]:


# Plotting data #
fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Actual Box Office')
ax.set_ylabel('Predicted Box Office')
plt.title('Actual vs. Predicted Box Office')
plt.ylim((50000000, 300000000)) 
plt.show()

