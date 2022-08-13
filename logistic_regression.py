import numpy as np
import matplotlib as plt
import pickle

from sklearn.metrics import precision_score, accuracy_score




import pandas as pd

df = pd.read_csv("framingham.csv").dropna()

# df = df.apply(lambda x: x/x.max(), axis=0)

# print(df.head())


x = df.drop("TenYearCHD", axis=1).values
y = df["TenYearCHD"].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

lrm = LogisticRegression2()
lrm.fit(x_train=x_train, y_train=y_train, epoch=100000, learning_rate=0.00015)