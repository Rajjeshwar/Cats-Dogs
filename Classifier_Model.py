#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    MaxPooling2D,
    Conv2D,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras import regularizers

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import numpy as np
from numpy import save
from numpy import load


# ### Load train and test

# In[2]:


def load_saved_data():
    X_train = load("X_train_cats-dogs.npy")
    Y_train = load("Y_train_cats-dogs.npy")
    X_test = load("X_test_cats-dogs.npy")
    Y_test = load("Y_test_cats-dogs.npy")

    return X_train, Y_train, X_test, Y_test


# In[3]:


X_train, Y_train, X_test, Y_test = load_saved_data()


# ### Train model with stratified k-fold cross validation

# In[4]:


cvscores = []


# In[5]:


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)


# In[6]:


for train, val in kfold.split(X_train, Y_train):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        X_train[train], Y_train[train], batch_size=32, epochs=30, verbose=1
    )

    scores = model.evaluate(X_train[val], Y_train[val])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)


# ### Analysis and metrics on train data

# In[7]:


print(cvscores)


# In[8]:


plt.plot(cvscores)


# In[9]:


print("mean: ", np.mean(cvscores))
print("std: ", np.std(cvscores))


# ### Evaluate on test set

# In[10]:


test_score = model.evaluate(X_test, Y_test)


# In[11]:


plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.legend(["accuracy", "loss"])
