from scipy.io.arff import loadarff 
from sklearn import metrics, datasets, tree 
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import mutual_info_classif 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Reading the ARFF file 
data = loadarff('pd_speech.arff') 
df = pd.DataFrame(data[0]) 
df['class'] = df['class'].str.decode('utf-8') 
 
predictor = tree.DecisionTreeClassifier() 
test_acc = [] 
train_acc = [] 
 
# 1. load and partition data 
X, y = df[list(df.columns[:-1])], df[["class"]] 
 
mutual_info = mutual_info_classif(X, y) 
array_mutual_info = np.array(mutual_info) 
 
for i in (5, 10, 40, 100, 250, 700): 
    # Get the top i values 
    indexes = np.argpartition(array_mutual_info, -i)[-i:] 
    topi_names = [] 
     
    # Select the top i columns 
    for j in range(i): 
        topi_names += [df.columns[indexes[j]]] 
    X = df[topi_names]  
     
    # Train with the selected features 
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, train_size = 
0.7, random_state = 1) 
     
    # Predict after training 
    predictor.fit(X_train, y_train) 
    y_pred = predictor.predict(X_test) 
    y_train_pred = predictor.predict(X_train) 
     
    # Metrics calculation 
    test_acc += [round(metrics.accuracy_score(y_test, y_pred), 2)] 
    train_acc += [round(metrics.accuracy_score(y_train, y_train_pred), 2)]

bar1 = np.arange(6) 
bar2 = [n + 0.1 for n in bar1] 
 
plt.bar(bar1, train_acc, color = "#7eb54e", label = "Training Accuracy") 
plt.bar(bar2, test_acc, color = "#ed9b4e", label = "Test Accuracy") 
 
plt.xticks([x + 0.1 for x in range(6)], [5, 10, 40, 100, 250, 700]) 
plt.xlabel("Number of features", fontsize = 15) 
plt.ylabel("Accuracy", fontsize = 15) 
plt.legend() 
plt.show()