import torch
import numpy as np
from layer import FeaturesEmbedding
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,log_loss
dataset_path = './data/'
training_set = np.load(dataset_path+'training_set.npy',allow_pickle=True)
valid_set = np.load(dataset_path+'validation_set.npy',allow_pickle=True)
test_set = np.load(dataset_path+'test_set.npy',allow_pickle=True)
X_train = []
y_train = []
X_test = []
y_test = []
train_mask = []
test_mask = []

for i in training_set:
    X_train.append(i[0])
    y_train.append(i[1])
    train_mask.append(i[3])
    
for i in valid_set:
    X_train.append(i[0])
    y_train.append(i[1])
    train_mask.append(i[3])

for i in test_set:
    X_test.append(i[0])
    y_test.append(i[1])
    test_mask.append(i[3])

#select the task

y_train = np.array(y_train)
y_test = np.array(y_test)

task_number = 2
y_train = y_train[:,task_number]
y_test = y_test[:,task_number]


X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
print(X_train.shape)
embedding  = FeaturesEmbedding(139, 30)
X_train = embedding(X_train)
X_test = embedding(X_test)
print(X_train.shape)

mask = [1,17,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,49]

for i in range(len(train_mask)):
    for j in range(len(train_mask[0])):
        if pd.isnull(train_mask[i][j]):
            X_train[i][mask[j]] = X_train[i][mask[j]] * train_mask[i][j]

for i in range(len(test_mask)):
    for j in range(len(test_mask[0])):
        if pd.isnull(test_mask[i][j]):
            X_test[i][mask[j]] = X_test[i][mask[j]] * test_mask[i][j]


X_train = X_train.view(len(X_train),-1).detach().numpy()
X_test = X_test.view(len(X_test),-1).detach().numpy()

print(X_train.shape)

# Define XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Train XGBoost model
xgb_model.fit(X_train, y_train)

# Make predictions on test set


y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test.reshape(-1, 1), y_pred_prob[:,1])
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("accuracy: ",accuracy)
print("auc score: ",auc)
print("precision score: ",precision)
print("f1_score: ",f1)

logloss = log_loss(y_test, y_pred_prob)

print("Log Loss: " , logloss)