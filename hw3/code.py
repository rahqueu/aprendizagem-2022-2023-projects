import warnings 
 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, learning_curve 
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.neural_network import MLPRegressor 
 
def warn(*args, **kwargs): 
    pass 
warnings.warn = warn 
 
# Reading the ARFF file 
data = loadarff('kin8nm.arff') 
df = pd.DataFrame(data[0]) 
X, y = df[list(df.columns[:-1])], df[["y"]] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)

rr = Ridge(alpha=0.1) 
mlpr_1 = MLPRegressor(hidden_layer_sizes=(10,10), activation="tanh", random_state=0, max_iter=500, early_stopping=True) 
mlpr_2 = MLPRegressor(hidden_layer_sizes=(10,10), activation="tanh", random_state=0, max_iter=500) 
 
rr.fit(X_train, y_train) 
mlpr_1.fit(X_train, y_train) 
mlpr_2.fit(X_train, y_train) 
 
y_test_rr = rr.predict(X_test) 
y_test_mlpr_1 = mlpr_1.predict(X_test) 
y_test_mlpr_2 = mlpr_2.predict(X_test) 
 
mae_rr = mean_absolute_error(y_test, y_test_rr) 
mae_mlpr_1 = mean_absolute_error(y_test, y_test_mlpr_1) 
mae_mlpr_2 = mean_absolute_error(y_test, y_test_mlpr_2) 
 
rr_residuals = [] 
mlpr_1_residuals = [] 
mlpr_2_residuals = [] 
 
size = len(y_test) 
for i in range(size): 
    rr_residuals.append(abs(y_test.iloc[i]["y"] - y_test_rr[i][0])) 
    mlpr_1_residuals.append(abs(y_test.iloc[i]["y"] - y_test_mlpr_1[i])) 
    mlpr_2_residuals.append(abs(y_test.iloc[i]["y"] - y_test_mlpr_2[i])) 
 
n_iteration_mlpr_1 = mlpr_1.niter 
n_iteration_mlpr_2 = mlpr_2.niter 
 
print("MLP1 iterations:", n_iteration_mlpr_1) 
print("MLP2 iterations:", n_iteration_mlpr_2) 
 
print("RMSE (Ridge):",np.sqrt(mean_squared_error(y_test,y_test_rr))) 
print("R2 (Ridge):", r2_score(y_test, y_test_rr)) 
 
print("RMSE (MLP1):",np.sqrt(mean_squared_error(y_test,y_test_mlpr_1))) 
print("R2 (MLP1):", r2_score(y_test, y_test_mlpr_1)) 
 
print("RMSE (MLP2):",np.sqrt(mean_squared_error(y_test,y_test_mlpr_2))) 
print("R2 (MLP2):", r2_score(y_test, y_test_mlpr_2)) 
 
plt.boxplot([rr_residuals, mlpr_1_residuals, mlpr_2_residuals], labels=["Ridge", "MLP1", "MLP2"]) 
plt.title("Residues (in absolute value) boxplot for each regression") 
plt.xlabel("Regressions") 
plt.ylabel("Residue value") 
plt.show() 
 
plt.hist(mlpr_2_residuals, edgecolor="black", linewidth=1, color="green", alpha=0.60, label="MLP2", align="mid") 
plt.hist(mlpr_1_residuals, edgecolor="black", linewidth=1, color="blue", alpha=0.60, label="MLP1", align="mid") 
plt.hist(rr_residuals, edgecolor="black", linewidth=1, color="red", alpha=0.60, label="Ridge", align="mid") 
 
plt.title("Residues (in absolute value) histogram for each regression") 
plt.xlabel("Residues") 
plt.ylabel("Frequency") 
plt.legend() 
plt.show() 
print("Ridge MAE:", mae_rr, "\nMLP1 MAE:", mae_mlpr_1, "\nMLP2 MAE:", mae_mlpr_2)