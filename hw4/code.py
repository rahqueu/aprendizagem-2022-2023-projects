from sklearn.preprocessing import MinMaxScaler 
import warnings 
import pandas as pd 
import numpy as np 
from scipy.io.arff import loadarff 
from sklearn import cluster, metrics 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

def warn(*args, **kwargs): 
    pass 
warnings.warn = warn 
 
# Reading the ARFF file 
data = loadarff('pd_speech.arff') 
df = pd.DataFrame(data[0]) 
df['class'] = df['class'].str.decode('utf-8') 
 
# Scale the dataframe 
scaler = MinMaxScaler() 
df = scaler.fit_transform(df) 
X_list = df[:, :-1] 
df = pd.DataFrame(df) 
 
X, y = df[list(df.columns[:-1])], df[[752]] 
 
temp_y = y.to_numpy() 
y_true = [int(x) for sublist in temp_y for x in sublist] 
 
kmeans = [] 
kmeans_model = [] 
silhouettes = [] 
purities = [] 
 
for i in range(3): 
    # Generate 3 KMeans clusterings with 3 different seeds (0, 1, 2) 
    kmeans.append(cluster.KMeans(n_clusters=3, random_state=i)) 
    kmeans_model.append(kmeans[i].fit(X)) 
 
    y_pred = kmeans_model[i].labels_ 
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred) 
 
    # Calculate silhouette and purity for the model 
    silhouette = metrics.silhouette_score(X, y_pred) 
    silhouettes.append(silhouette) 
    print("Silhouette", str(i) + ":", silhouette) 
    purity = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 
    purities.append(purity) 
    print("Purity", str(i) + ":", purity) 
 
# Fix random = 0 
y_pred = kmeans_model[0].labels_ 
 
# Get the indexes for the 2 features with the biggest variances 
variances = X.var().to_numpy() 
indexes = np.argpartition(variances, -2)[-2:]

scatter_X = X_list[:, indexes[0]] 
scatter_Y = X_list[:, indexes[1]] 
 
fig = plt.figure() 
ax1 = fig.add_subplot(121) 
ax2 = fig.add_subplot(122) 
 
ax1.set_title("Scatter plot of the original diagnoses") 
for g in np.unique(y_true): 
    # Select the indexes where we find the specified label 
    ix = np.where(y_true == g) 
    ax1.scatter(scatter_X[ix], scatter_Y[ix], label=g) 
ax1.set_xlabel(X.columns[indexes[0]]) 
ax1.set_ylabel(X.columns[indexes[1]]) 
 
ax2.set_title("Scatter plot of the k=3 clusters") 
for g in np.unique(y_pred): 
    ix = np.where(y_pred == g) 
    ax2.scatter(scatter_X[ix], scatter_Y[ix], label=g) 
ax2.set_xlabel(X.columns[indexes[0]]) 
ax2.set_ylabel(X.columns[indexes[1]]) 
plt.legend() 
plt.show() 
 
# Calculate number of primary components needed 
components = 0 
size = len(X_list[0]) 
for i in range(size): 
    pca = PCA(n_components=i) 
    pca.fit(X) 
    if sum(pca.explained_variance_ratio_) > 0.8: 
        components = i 
        break 
 
print("Number of primary components to explain 80% variability:", components)