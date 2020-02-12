import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
import keras
from keras.models import Sequential
from keras.layers import Dense

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
print(concrete_data.describe())

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']].copy() # all columns except Strength
target = concrete_data['Strength'].copy() # Strength column

n_cols = predictors.shape[1]

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

mse_dict = {}

for i in range(50):
    X = predictors.copy()
    y = target.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = regression_model()
    model.fit(X_train, y_train, epochs = 50, verbose=0)
    mse_dict['%s' % (i)] = model.evaluate(X_test, y_test, verbose=0)

print(mse_dict)

mse_mean = np.mean(list(mse_dict.values()))
mse_std = np.std(list(mse_dict.values()))
print('mean: ', mse_mean)
print('std: ', mse_std)

#part B
# Normalize (standardize) the data
predictors_b = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']].copy() # all columns except Strength
target_b = concrete_data['Strength'].copy() # Strength column
predictors_norm = (predictors_b - predictors_b.mean()) / predictors_b.std()
print(predictors_norm.head())

mse_dict_b = {}

for i in range(50):
    X = predictors_norm.copy()
    y = target_b.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = regression_model()
    model.fit(X_train, y_train, epochs = 50, verbose=0)
    mse_dict_b['%s' % (i)] = model.evaluate(X_test, y_test, verbose=0)

print(mse_dict_b)

print(mse_dict_b)

mse_mean = np.mean(list(mse_dict_b.values()))
mse_std = np.std(list(mse_dict_b.values()))

print('mean: ', mse_mean)
print('std: ', mse_std)

# part c
mse_dict_c = {}

for i in range(50):
    X = predictors_norm.copy()
    y = target_b.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = regression_model()
    model.fit(X_train, y_train, epochs = 100, verbose=0)
    mse_dict_c['%s' % (i)] = model.evaluate(X_test, y_test, verbose=0)

print(mse_dict_c)

mse_mean = np.mean(list(mse_dict_c.values()))
mse_std = np.std(list(mse_dict_c.values()))
print('mean: ', mse_mean)
print('std: ', mse_std)

# part D

# define new regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


mse_dict_d = {}

for i in range(50):
    X = predictors_norm.copy()
    y = target_b.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = regression_model()
    model.fit(X_train, y_train, epochs = 50, verbose=0)
    mse_dict_d['%s' % (i)] = model.evaluate(X_test, y_test, verbose=0)

print(mse_dict_d)

mse_mean = np.mean(list(mse_dict_d.values()))
mse_std = np.std(list(mse_dict_d.values()))
print('mean: ', mse_mean)
print('std: ', mse_std)


# mean_squared_error_list.append(history.history['mean_squared_error'][len(history.history['mean_squared_error'])-1])






































