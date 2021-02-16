# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",
                  strategy = "mean",axis=0) #use mean, median or most_frequent
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#but now the ols will assume that 2>1 in the category column viz 
#mathematically true but not a good idea for our model
#so we use another - onehotencoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#creates diff arrays^
#since y is DV, we only use labelencoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting into training and test data
#smaller test size recommended
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.25,
                                                    random_state = 0) 
'''
feature scaling
used when two variables have different scales
will cause error in model due to euclidean distance
the euclidean distance will be dominated by the one with 
higher range. To avoid that, you should either use
normalisation or standardisation.'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#fit and transform the training set.
#only transform the test set
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#scaling categorical data depends on dataset
'''
It depends on how much you wanna keep interpretation
in your models because if you scale this, it will be good 
because everything will be on the same scale, and you will be happy with 
that and it will be good for your predictions but you will lose the 
interpretation of knowing which observations belongs to which country, et cetera.
'''
