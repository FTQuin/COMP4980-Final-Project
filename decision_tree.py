import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, random_split

from SeasonDataset import SeasonDataset

# CONSTANTS
DATA_FILE = './data/f_and_a_no0GP_-1.csv'
SAVE_FILE = 'models/dt.pickle'
TEST_PERCENT = 0.2

# get dataset
sd = SeasonDataset(DATA_FILE)
sd.df_features.reset_index(drop=True)
indicies = sd.df_features.index.values.tolist()

train_X, test_X, train_y, test_y = train_test_split(indicies, sd.df_features[sd.label], test_size=TEST_PERCENT)
train_X = sd.df_features.iloc[train_X].drop(sd.label+sd.one_hot_features, axis=1)
test_X = sd.df_features.iloc[test_X].drop(sd.label+sd.one_hot_features, axis=1)

# instantiate dt
dt = DecisionTreeClassifier(max_depth=1000, criterion='entropy', random_state=1)

# Fit dt to train set
dt.fit(train_X, train_y)

# get accuracy on test set
y_pred = dt.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)
print(accuracy)

# save dt
with open(SAVE_FILE, 'wb') as file:
    pickle.dump(dt, file, protocol=pickle.HIGHEST_PROTOCOL)
