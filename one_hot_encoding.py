import numpy as np
from sympy import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def get_varlist(df):
  # creating variables for all xs using symbols sympy
  x_s = []
  for col in df.columns:
      x_s.append(col)
  # print(x_s)
  return x_s

def transform_label_encoding(df):
  df[get_varlist(df)] = df[get_varlist(df)].apply(LabelEncoder().fit_transform)
  return df

def transform_oneHotEncoding(df):
  df = pd.get_dummies(df, columns=get_varlist(df), dtype=float )
  return df

def logistic_regression_accuracy_on(dataframe, encoding_type):
    
    le = LabelEncoder()
    dataframe['y'] = le.fit_transform(df['y'])
    y = dataframe["y"]
    X = dataframe.drop("y", axis=1)

    if encoding_type=="label":
      X = transform_label_encoding(X)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
      # print(X_train.head())
      # print(X_test.head())
      logit = LogisticRegression(max_iter=10000)
      logit.fit((X_train), y_train)
      print (classification_report(y_test, logit.predict((X_test))))

    elif encoding_type=="onehot":
      X = transform_oneHotEncoding(X)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
      # print(X_train.head())
      # print(X_test.head())
      logit = LogisticRegression(max_iter=10000)
      logit.fit((X_train), y_train)
      print (classification_report(y_test, logit.predict((X_test))))

df = pd.read_csv("./bank+marketing/bank/bank.csv", sep=";")

print("Regression on dataset with label encoding:")
print(logistic_regression_accuracy_on(df, "label"))
print("Regression on dataset with one hot encoding:")
print(logistic_regression_accuracy_on(df, "onehot"))