# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # remember 13 is not included
y = dataset.iloc[:, 13].values

# Encoding categorical data     
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encoding one by one - getting rid of strings - instead getting 0 or 1 or 2 or etc
labelencoder_X_1 = LabelEncoder()   # creating the object
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # using the method in the obejct
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# because the categorical variable are not ordinal,
# there is no relatioinal order between them (France is not higher than Germany etc),
# we have to create dummy variable

# we don't need to create it for gender, because we would remove one to avoid
# dummy variable trap anyway
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# removing one dummy variable to avoid dummy variable trap
X = X[:, 1:] # taking all the rows and all the columns except the first one with index 0

# y that is 0,1 is not necessary, the strings are problem

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling   # everytime in Deep learning !!
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!  - applied dropout for case of overfitting

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # required to initialize the ANN
from keras.layers import Dense # required to build the layers of the ANN
                               # dense function from this model takes care of initializing small weights at the beginning
from keras.layers import Dropout #in case of overfitting

# Initialising the ANN
classifier = Sequential() # creating the object

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# units can be average of nodes in input layer and output layer 11+1=12/2=6 or parameter tunning
# kernel_initializer = uniform makes all the weights according to uniform distribution, close to 0,
# activation = function we want to choose rectifier activation function - 'relu'
# input_dim = 11-number of variables, because this is the first layer we are creating
classifier.add(Dropout(p = 0.1))
# p how many neurons should be dropped..start 0.1 (10 % are dropped), if not help go for 0.2 etc

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# units is number of output nodes = 1 variable
# activation function is sigmoid function to get the probabilities
# if we have 3 categories as output => units = 3, activation = softmax, which is sigmoid function,
# applied to variables with 3 and more categories

# Compiling the ANN = applying stochastical gradient decent on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer = algorithm we want to use to find the best weights = stochastical gradient decent - adam is a type of it,
# loss = loss function within this algorithm, similar as sum of square in regression, it is a logaritmic loss function,
# in case we have 3 and more categories => categorical_crossentropy
# metrics = criterium to evaluate the model, we are adding the brackets because it is expecting a list

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# batch_size = number of inputs after which we adjust the weights - parameter tunning
# epoch = applying steps 1-6 - parameter tunning

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # changing probabilities to TRUE and FALSE

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# computing accuracy
(1517 + 203)/2000

# Predicting on a single new observation 
"""Predict if the customer with the following informations will leave the bank:
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40
    Tenure: 3
    Balance: 60000
    Number of Products: 2
    Has Credit Card: Yes
    Is Active Member: Yes
    Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0 ,0 ,600 ,1 , 40, 3, 60000, 2, 1, 1, 50000]])))
#to create a horizontal array use [[]] to create a two dimensional array, otherwise [] will make a column out of it
# we have to use the same scaling set that was fitted to the training set 
new_prediction = (new_prediction > 0.5)

# Part 4 - Evaluating, Improving and Tuning the ANN
# rerunning the importing and preprocessing part

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
# we build ANN with Keras but k-fold cross validation is in scikit,
# so we need the keras wrapper to combine them
from sklearn.model_selection import cross_val_score
# importing the cross validation function

from keras.models import Sequential 
from keras.layers import Dense

def build_classifier(): # function that builds the architecture of ANN = the classifier
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# creating a classifier trained using k-fold cross validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#if __name__ == "__main__":
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed
# In case of overfitting (high variance) we drop some neurons in each iteration so,
# it can't learn too much.. see part 2

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer): # add it as an argument so we can tune it
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [20, 25],
              'epochs': [500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           verbose = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_