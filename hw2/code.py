import warnings 
from sklearn import metrics, datasets, tree 
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from scipy import stats 
from scipy.io.arff import loadarff 
from sklearn.naive_bayes import GaussianNB 
import seaborn as sns 
from sklearn.preprocessing import normalize 
 
def warn(*args, **kwargs): 
    pass 
warnings.warn = warn 
 
# Reading the ARFF file 
data = loadarff('pd_speech.arff') 
df = pd.DataFrame(data[0]) 
df['class'] = df['class'].str.decode('utf-8') 
X, y = df[list(df.columns[:-1])], df[["class"]] 
 
X = normalize(X) 
X = pd.DataFrame(X) 
 
predictor_NB = GaussianNB() 
predictor_kNN = KNeighborsClassifier(n_neighbors=5, p=2, weights="uniform") 
 
folds_acc_NB = [] 
folds_acc_kNN = [] 
overfit_NB = [] 
overfit_kNN = []

total_confusion_NB = np.array(((0, 0), (0, 0))) 
total_confusion_kNN = np.array(((0, 0), (0, 0))) 
folds = StratifiedKFold(n_splits=10, random_state=0, shuffle=True) 
 
# 0 = Sick; 1 = Healthy 
for train_k, test_k in folds.split(X, y): 
 
    X_train, X_test = X.iloc[train_k], X.iloc[test_k] 
    y_train, y_test = y.iloc[train_k], y.iloc[test_k] 
 
    predictor_NB.fit(X_train, y_train) 
    y_pred_NB = predictor_NB.predict(X_test) 
    cm_NB = np.array(confusion_matrix(y_test, y_pred_NB, labels=["0", "1"])) 
    folds_acc_NB.append(round(metrics.accuracy_score(y_test, y_pred_NB), 2)) 
    y_pred_NB = predictor_NB.predict(X_train) 
    overfit_NB.append(round(metrics.accuracy_score(y_train, y_pred_NB), 2)) 
    total_confusion_NB = np.add(total_confusion_NB, cm_NB) 
 
    predictor_kNN.fit(X_train, y_train) 
    y_pred_kNN = predictor_kNN.predict(X_test) 
    cm_kNN = np.array(confusion_matrix(y_test, y_pred_kNN, labels=["0", "1"])) 
    folds_acc_kNN.append(round(metrics.accuracy_score(y_test, y_pred_kNN), 2)) 
    y_pred_kNN = predictor_kNN.predict(X_train) 
    overfit_kNN.append(round(metrics.accuracy_score(y_train, y_pred_kNN), 2)) 
    total_confusion_kNN = np.add(total_confusion_kNN, cm_kNN) 
 
confusion_NB = pd.DataFrame(total_confusion_NB, index=["Sick", "Healthy"], columns=["Predicted Sick", "Predicted Healthy"]) 
confusion_kNN = pd.DataFrame(total_confusion_kNN, index=["Sick", "Healthy"], columns=["Predicted Sick", "Predicted Healthy"]) 
 
heat = sns.heatmap(confusion_NB, annot=True, fmt='g') 
plt.title("Cumulative Naïve-Bayes confusion matrix") 
plt.show() 
heat2 = sns.heatmap(confusion_kNN, annot=True, fmt='g') 
plt.title("Cumulative kNN confusion matrix") 
plt.show() 
 
classifiers = ( 
    ("Naive Bayes", predictor_NB), 
    ("kNN", predictor_kNN) 
) 
 
print("Overfit NB?\nTraining accuracy:", round(sum(overfit_NB)/len(overfit_NB), 2), "\nTesting accuracy:", round(sum(folds_acc_NB)/len(folds_acc_NB), 2)) 
print("Overfit kNN?\nTraining accuracy:", round(sum(overfit_kNN)/len(overfit_kNN), 2), "\nTesting accuracy:", round(sum(folds_acc_kNN)/len(folds_acc_kNN), 2)) 
 
for name, classifier in classifiers: 
    accs = cross_val_score(classifier, X, y, cv=10, scoring='accuracy') 
    print(name, "accuracy =", round(np.mean(accs), 2), "±", round(np.std(accs), 2)) 
 
# NB > kNN? 
res = stats.ttest_rel(folds_acc_NB, folds_acc_kNN, alternative='greater') 
print("p1>p2? pval=", res.pvalue) 
# NB < kNN? 
res = stats.ttest_rel(folds_acc_NB, folds_acc_kNN, alternative='less') 
print("p1<p2? pval=", res.pvalue) 
# NB != kNN? 
res = stats.ttest_rel(folds_acc_NB, folds_acc_kNN, alternative='two-sided') 
print("p1!=p2? pval=", res.pvalue)