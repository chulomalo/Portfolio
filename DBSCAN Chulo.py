#!/usr/bin/env python
# coding: utf-8

# # Practical Case: DBSCAN
# 

# ### Description
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. 
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# ### Find the Dataset
# You can access the dataset [here](https://drive.google.com/file/d/1VR84Ima2CFIBPIu6d82ZjRlD3DqZwksL/view?usp=drive_link).
# 

# ## Imports

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import metrics
import numpy as np
import math


# ## Auxiliar Functions 

# In[2]:


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# ## 1. Reading the Dataset

# In[4]:


df = pd.read_csv("/Users/sergioperez/Desktop/Data/udemy ml/datasets/creditcard.csv")


# ## 2. Visualizing the Dataset

# In[4]:


df.head(10)


# In[5]:


print("Number of columns:", len(df.columns))
print("Dataset's length :", len(df))


# In[6]:


# 492 fraudulent transactions, 284,315 legitimate transactions
# The dataset is invalanced

df["Class"].value_counts()


# In[7]:


# Here we can visualize each one of the attributes
df.info()


# In[8]:


# Looking for null values
df.isna().any()


# In[9]:


df.describe()


# In[10]:


# Drop the 'Class' column and get the remaining features
features = df.drop("Class", axis=1)

# Define the number of columns you want in the plot grid
n_cols = 4
n_features = len(features)
max_rows = 8  # Set a maximum number of rows to avoid an overly large figure
n_rows = min(math.ceil(n_features / n_cols), max_rows)  # Limit the number of rows

# Set up the figure with a larger size for better visibility
plt.figure(figsize=(n_cols * 5, n_rows * 5))  # Adjust the width and height as needed
gs = gridspec.GridSpec(n_rows, n_cols)

# Plot each feature's distribution for fraudulent and legitimate transactions
for i, f in enumerate(features[:n_rows * n_cols]):  # Limit to first n_rows * n_cols features
    ax = plt.subplot(gs[i])
    sns.kdeplot(df[f][df["Class"] == 1], label='Fraudulent', fill=True, ax=ax)
    sns.kdeplot(df[f][df["Class"] == 0], label='Legitimate', fill=True, ax=ax)
    ax.set_title(f'Feature: {f}')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

# Manually adjust the padding
plt.subplots_adjust(hspace=1.5, wspace=0.5)  # Manually adjust the space between plots
plt.show()


# In[11]:


# Graphical representation of two features
plt.figure(figsize=(12, 6))  # Set the figure size
plt.scatter(df["V10"][df['Class'] == 0], df["V14"][df['Class'] == 0], c="g", marker=".")  # Plot legitimate transactions (Class 0) in green
plt.scatter(df["V10"][df['Class'] == 1], df["V14"][df['Class'] == 1], c="r", marker=".")  # Plot fraudulent transactions (Class 1) in red
plt.xlabel("V10", fontsize=14)  # Set the x-axis label to 'V10'
plt.ylabel("V14", fontsize=14)  # Set the y-axis label to 'V14'
plt.show()  # Display the plot


# ## 3. Data Preparation
# For this type of algorithm, it is important that all the data is within a similar range. Therefore, we can apply a scaling or normalization function. Another option is to remove the features that are not within a similar range, as long as they are not very influential for the prediction.

# In[12]:


df = df.drop(["Time", "Amount"], axis=1)


# ## 4. DBSCAN with a Two-Dimensional Dataset
# Before starting the DBSCAN training on all the dataset attributes, a test is performed on two attributes to understand how it constructs the decision boundary.

# In[13]:


X = df[["V10", "V14"]].copy()
y = df["Class"].copy()


# In[14]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.15, min_samples=13)
dbscan.fit(X)


# In[15]:


def plot_dbscan(dbscan, X, size):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker=".", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


# In[16]:


plt.figure(figsize=(12, 6))
plot_dbscan(dbscan, X.values, size=100)
plt.xlabel("V10", fontsize=14)
plt.ylabel("V14", fontsize=14)
plt.show()


# In[17]:


counter = Counter(dbscan.labels_.tolist())
bad_counter = Counter(dbscan.labels_[y == 1].tolist())

for key in sorted(counter.keys()):
    print("Label {0} has {1} samples - {2} are malicious samples".format(
        key, counter[key], bad_counter[key]))


# ## 5. Column reduction

# In[26]:


X = df.drop("Class", axis=1)
y = df["Class"].copy()


# ### We apply Column selection with Random Forest
# 
# 
# 
# 
# 
# 

# In[19]:


from sklearn.ensemble import RandomForestClassifier

clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X, y)


# In[24]:


# Choosing the most relevent for our analysis
feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)


# In[23]:


# Now we reduce the dataset to the 7 most relevant columns
X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()


# In[25]:


X_reduced


# ## DBS Training with the reduced dataset

# In[28]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.70, min_samples=25)
dbscan.fit(X_reduced)


# In[29]:


counter = Counter(dbscan.labels_.tolist())
bad_counter = Counter(dbscan.labels_[y == 1].tolist())

for key in sorted(counter.keys()):
    print("Label {0} has {1} samples - {2} are malicious samples".format(
        key, counter[key], bad_counter[key]))


# ## 6. Evaluating the results

# I was careful with imbalanced datasets when using metrics to measure the purity of the clusters. One possible solution is to use techniques for balancing the dataset, such as generating more examples of fraudulent transactions or reducing the number of legitimate transaction examples.

# In[30]:


clusters = dbscan.labels_


# In[32]:


# Calculating the Purity Score.

print("Purity Score:", purity_score(y, clusters))


# In[35]:


# Calculating the Shiloutte Score.

print("Shiloutte: ", metrics.silhouette_score(X_reduced, clusters, sample_size=10000))


# In[37]:


# Calculating the Calinski harabasz.

print("Calinski harabasz: ", metrics.calinski_harabasz_score(X_reduced, clusters))


# ## 7. Typical Problems That Can Be Solved with DBSCAN
# DBSCAN is useful for solving problems like the one presented below.

# In[41]:


#Generating a Dataset

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)


# In[42]:


plt.figure(figsize=(12, 6))
plt.scatter(X[:,0][y == 0], X[:,1][y == 0], c="g", marker=".")
plt.scatter(X[:,0][y == 1], X[:,1][y == 1], c="r", marker=".")
plt.show()


# In[43]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.1, min_samples=6)
dbscan.fit(X)


# In[45]:


# Representing the Desicion Limit
plt.figure(figsize=(12, 6))
plot_dbscan(dbscan, X, size=100)
plt.show()


# In[46]:


counter = Counter(dbscan.labels_.tolist())
bad_counter = Counter(dbscan.labels_[y == 1].tolist())

for key in sorted(counter.keys()):
    print("Label {0} has {1} samples - {2} are malicious samples".format(
        key, counter[key], bad_counter[key]))

