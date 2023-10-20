import tensorflow_hub as hub

import pandas as pd

import tensorflow_text as text

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

import numpy as np

# load data
df = pd.read_csv('spam_data_.csv')

#df.drop(columns=[""])
print(df.head())

# check count and unique and top values and their frequency
print(df['Category'].value_counts())

# check percentange of data - states how much data needs to be balanced
print(str(round(747/4825,2))+'%')

# creating 2 new dataframe as df_ham , df_spam

df_spam = df[df['Category']=='spam']
print("Spam Dataset Shape:", df_spam.shape)

df_ham = df[df['Category']=='ham']
print("Ham Dataset Shape:", df_ham.shape)

# downsampling ham dataset - take only random 747 example
# will use df_spam.shape[0] - 747

df_ham_downsampled = df_ham.sample(df_spam.shape[0])
print(df_ham_downsampled.shape)

# concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
df_balanced = pd.concat([df_spam , df_ham_downsampled])
print(df_balanced.head())

print(df_balanced['Category'].value_counts())

print(df_balanced.sample(10))

# creating numerical repersentation of category - one hot encoding
df_balanced['spam'] = df_balanced['Category'].apply(lambda x:1 if x=='spam' else 0)

# displaying data - spam -1 , ham-0
print(df_balanced.sample(4))

# loading train test split
from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(df_balanced['Message'], df_balanced['spam'], stratify = df_balanced['spam'])

# check for startification
print(y_train.value_counts())

print(560/560)

print(y_test.value_counts())

print(187/187)