#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# Welcome to "***Clustering (Customer Segmentation) Project***". This is the last medium project of ***Machine Learning*** course. 
# 
# At the end of this project, you will have performed ***Cluster Analysis*** with an ***Unsupervised Learning*** method.
# 
# ---
# 
# In this project, customers are required to be segmented according to the purchasing history obtained from the membership cards of a big mall.
# 
# This project is less challenging than other projects. After getting to know the data set quickly, you are expected to perform ***Exploratory Data Analysis***. You should observe the distribution of customers according to different variables, also discover relationships and correlations between variables. Then you will spesify the different variables to use for cluster analysis.
# 
# The last step in customer segmentation is to group the customers into distinct clusters based on their characteristics and behaviors. One of the most common methods for clustering is ***K-Means Clustering***, which partitions the data into k clusters based on the distance to the cluster centroids. Other clustering methods include ***hierarchical clustering***, density-based clustering, and spectral clustering. Each cluster can be assigned a label that describes its main features and preferences.
# 
# - ***NOTE:*** *This project assumes that you already know the basics of coding in Python. You should also be familiar with the theory behind Cluster Analysis and scikit-learn module as well as Machine Learning before you begin.*

# ---
# ---

# # #Tasks

# Mentoring Prep. and self study#### 
# 
# #### 1. Import Libraries, Load Dataset, Exploring Data
# - Import Libraries
# - Load Dataset
# - Explore Data
# 
# #### 2. Exploratory Data Analysis (EDA)
# 
# 
# #### 3. Cluster Analysis
# 
# - Clustering based on Age and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Clustering based on Annual Income and Spending Score
# 
#     *i. Create a new dataset with two variables of your choice*
#     
#     *ii. Determine optimal number of clusters*
#     
#     *iii. Apply K Means*
#     
#     *iv. Visualizing and Labeling All the Clusters*
#     
#     
# - Hierarchical Clustering
# 
#     *i. Determine optimal number of clusters using Dendogram*
# 
#     *ii. Apply Agglomerative Clustering*
# 
#     *iii. Visualizing and Labeling All the Clusters* 
# 
# - Conclusion

# ---
# ---

# ## 1. Import Libraries, Load Dataset, Exploring Data
# 
# There is a big mall in a specific city that keeps information of its customers who subscribe to a membership card. In the membetrship card they provide following information : gender, age and annula income. The customers use this membership card to make all the purchases in the mall, so tha mall has the purchase history of all subscribed members and according to that they compute the spending score of all customers. You have to segment these customers based on the details given. 

# #### Import Libraries

# In[53]:


from sklearn.metrics import classification_report, confusion_matrix,  average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


# #### Load Dataset

# In[54]:


df0 = pd.read_csv('Mall_Customers.csv')
df = df0.copy()
df.head()


# #### Explore Data
# 
# You can rename columns to more usable, if you need.

# In[55]:


df.info()


# In[56]:


df.rename(columns={"Annual Income (k$)":"Annual_Income",
          'Spending Score (1-100)': 'Spending_Score'}, inplace=True)


# In[57]:


df.shape


# In[58]:


df.isnull().sum()


# In[59]:


df.duplicated().sum()


# ---
# ---

# ## 2. Exploratory Data Analysis (EDA)
# 
# After performing Cluster Analysis, you need to know the data well in order to label the observations correctly. Analyze frequency distributions of features, relationships and correlations between the independent variables and the dependent variable. It is recommended to apply data visualization techniques. Observing breakpoints helps you to internalize the data.
# 
# 
# 
# 

# In[60]:


df.head()


# In[61]:


df.describe()


# In[62]:


print('Mean of Age:', df.Age.mean())
print('Min of Age:', df.Age.min())
print('Max of Age:', df.Age.max())


# In[63]:


df[(df['Spending_Score'] == df['Spending_Score'].max())]


# In[64]:


df[(df['Annual_Income'] == df['Annual_Income'].max())]


# In[65]:


df.CustomerID.nunique()


# In[66]:


df.drop('CustomerID', axis=1, inplace=True)


# In[67]:


df.head()


# In[68]:


df_num = df.select_dtypes(include=np.number)
df_num.head()


# In[69]:


df_num.corr()


# In[70]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, showmeans=True,
            meanprops={"marker": "o",
                       "markerfacecolor": "white",
                       "markeredgecolor": "black",
                       "markersize": "10"})
plt.show()


# In[71]:


gender_income = df[['Gender', 'Annual_Income']].groupby('Gender', as_index = False).mean()
gender_income


# In[72]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=gender_income['Gender'], y=gender_income['Annual_Income'], alpha=0.8, ax=ax)
ax.set_title('Annual Income by Gender')
ax.set_ylabel('Mean Annual Income', fontsize=12)
ax.set_xlabel('Gender', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

plt.show()


# In[73]:


gender_spending = df[['Gender', 'Spending_Score']].groupby('Gender', as_index = False).mean()
gender_spending


# In[74]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=gender_spending['Gender'], y=gender_spending['Spending_Score'], alpha=0.8, ax=ax)
ax.set_title('Spending Score by Gender')
ax.set_ylabel('Mean Spending Score', fontsize=12)
ax.set_xlabel('Gender', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

plt.show()


# In[75]:


gender_age = df[['Gender', 'Age']].groupby('Gender', as_index = False).mean()
gender_age


# In[76]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=gender_age['Gender'], y=gender_age['Age'], alpha=0.8, ax=ax)
ax.set_title('Mean Age by Gender')
ax.set_ylabel('Mean Age of the Customer', fontsize=12)
ax.set_xlabel('Gender', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

plt.show()


# In[77]:


fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Gender')
plt.show()


# In[78]:


fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(data=df, x='Age', y='Spending_Score', hue='Gender')
plt.show()


# In[79]:


sns.pairplot(df, hue='Gender')
plt.show()


# In[ ]:





# ---
# ---

# ## 3. Cluster Analysis

# The purpose of the project is to perform cluster analysis using [K-Means](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1) and [Hierarchical Clustering](https://medium.com/analytics-vidhya/hierarchical-clustering-d2d92835280c) algorithms.
# Using a maximum of two variables for each analysis can help to identify cluster labels more clearly.
# The K-Means algorithm requires determining the number of clusters using the [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering), while Hierarchical Clustering builds a dendrogram without defining the number of clusters beforehand. Different labeling should be done based on the information obtained from each analysis.
# Labeling example: 
# 
# - **Normal Customers**  -- An Average consumer in terms of spending and Annual Income
# - **Spender Customers** --  Annual Income is less but spending high, so can also be treated as potential target customer.

# ### Clustering based on Age and Spending Score

# #### *i. Create a new dataset with two variables of your choice*

# In[80]:


df1 = df.drop(columns="Gender")


# In[81]:


from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd


def hopkins(data_frame, sampling_size):
    """Assess the clusterability of a dataset. A score between 0 and 1, a score around 0.5 express
    no clusterability and a score tending to 0 express a high cluster tendency.

    Parameters
    ----------
    data_frame : numpy array
        The input dataset
    sampling_size : int
        The sampling size which is used to evaluate the number of DataFrame.

    Returns
    ---------------------
    score : float
        The hopkins score of the dataset (between 0 and 1)

    Examples
    --------
    >>> from sklearn import datasets
    >>> from pyclustertend import hopkins
    >>> X = datasets.load_iris().data
    >>> hopkins(X,150)
    0.16
    """

    if type(data_frame) == np.ndarray:
        data_frame = pd.DataFrame(data_frame)

    # Sample n observations from D : P

    if sampling_size > data_frame.shape[0]:
        raise Exception(
            'The number of sample of sample is bigger than the shape of D')

    data_frame_sample = data_frame.sample(n=sampling_size)

    # Get the distance to their neirest neighbors in D : X

    tree = BallTree(data_frame, leaf_size=2)
    dist, _ = tree.query(data_frame_sample, k=2)
    data_frame_sample_distances_to_nearest_neighbours = dist[:, 1]

    # Randomly simulate n points with the same variation as in D : Q.

    max_data_frame = data_frame.max()
    min_data_frame = data_frame.min()

    uniformly_selected_values_0 = np.random.uniform(min_data_frame[0], max_data_frame[0], sampling_size)
    uniformly_selected_values_1 = np.random.uniform(min_data_frame[1], max_data_frame[1], sampling_size)

    uniformly_selected_observations = np.column_stack((uniformly_selected_values_0, uniformly_selected_values_1))
    if len(max_data_frame) >= 2:
        for i in range(2, len(max_data_frame)):
            uniformly_selected_values_i = np.random.uniform(min_data_frame[i], max_data_frame[i], sampling_size)
            to_stack = (uniformly_selected_observations, uniformly_selected_values_i)
            uniformly_selected_observations = np.column_stack(to_stack)

    uniformly_selected_observations_df = pd.DataFrame(uniformly_selected_observations)

    # Get the distance to their neirest neighbors in D : Y

    tree = BallTree(data_frame, leaf_size=2)
    dist, _ = tree.query(uniformly_selected_observations_df, k=1)
    uniformly_df_distances_to_nearest_neighbours = dist

    # return the hopkins score

    x = sum(data_frame_sample_distances_to_nearest_neighbours)
    y = sum(uniformly_df_distances_to_nearest_neighbours)

    if x + y == 0:
        raise Exception('The denominator of the hopkins statistics is null')

    return x / (x + y)[0]


# In[82]:


hopkins(df1, df1.shape[0])


# In[83]:


X1=df[['Age','Spending_Score']]
X1.head()


# In[84]:


X1.shape


# In[85]:


hopkins(X1, X1.shape[0])


# In[ ]:





# #### *ii. Determine optimal number of clusters*

# In[86]:


from sklearn.cluster import KMeans


# In[87]:


ssd = []

K = range(2,11)

for k in K:
    model = KMeans(n_clusters = k, random_state = 42)
    model.fit(X1)
    ssd.append(model.inertia_)


# In[88]:


plt.figure(figsize=(10, 6))
plt.plot(K, ssd, "bo--")
plt.xlabel("Different k values")
plt.ylabel("ssd") 
plt.title("The Elbow Method") 
plt.show()


# In[89]:


ssd


# In[90]:


-pd.Series(ssd).diff()


# In[91]:


from yellowbrick.cluster import KElbowVisualizer

model_ = KMeans(random_state=42)
visualizer = KElbowVisualizer(model_, k=(2, 9), size=(1080, 720))

visualizer.fit(X1)        
visualizer.poof();


# In[92]:


from sklearn.metrics import silhouette_score


# In[93]:


range_n_clusters = range(2,11)
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X1)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X1, cluster_labels)
    print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")


# In[ ]:





# In[ ]:





# ### Why silhouette_score is negative?

# ![image.png](attachment:image.png)
silhouette_score = (b-a)/max(a,b)

b : the mean nearest-cluster distance 
a : the mean intra-cluster distance 

for red point, 

b = 1 
a = ((1+1)**0.5 + (1+1)**0.5)/2  ==> 1.41

silhouette_score = (1-1.41)/1.41 ==> -0.29
# #### *iii. Apply K Means*

# In[94]:


from sklearn.cluster import KMeans


# In[95]:


from yellowbrick.cluster import SilhouetteVisualizer

model_ssd = KMeans(n_clusters=4, random_state=42)
visualizer = SilhouetteVisualizer(model_ssd, size=(1080, 720))

visualizer.fit(X1)    # Fit the data to the visualizer
visualizer.poof()


# In[96]:


for i in range(model_ssd.n_clusters):
    label = (model_ssd.labels_ == i)
    print(f"mean silhouette score for label {i:<4} : {visualizer.silhouette_samples_[label].mean()}")
print(f"mean silhouette score for all labels : {visualizer.silhouette_score_}")


# In[ ]:





# #### *iv. Visualizing and Labeling All the Clusters*

# In[ ]:





# In[ ]:





# In[ ]:





# ### Clustering based on Annual Income and Spending Score

# #### *i. Create a new dataset with two variables of your choice*

# In[100]:


X2=df[['Annual_Income','Spending_Score']]
X2.head()


# In[101]:


X2.shape


# In[102]:


hopkins(X2, X2.shape[0])


# #### *ii. Determine optimal number of clusters*

# In[103]:


ssd = []

K = range(2,11)

for k in K:
    kmeans = KMeans(n_clusters =k, random_state=42)
    kmeans.fit(X2)
    ssd.append(kmeans.inertia_)


# In[104]:


plt.figure(figsize=(12, 8))
plt.plot(K, ssd, "bo--")
plt.xlabel("Different k values")
plt.ylabel("ssd") 
plt.title("The Elbow Method");


# In[105]:


ssd


# In[106]:


-pd.Series(ssd).diff()


# In[107]:


from yellowbrick.cluster import KElbowVisualizer

model_ = KMeans(random_state=42)
visualizer = KElbowVisualizer(model_, k=(2, 10), size=(1080, 720))

visualizer.fit(X2)        # Fit the data to the visualizer
visualizer.show();


# In[108]:


range_n_clusters = range(2,10)
for num_clusters in range_n_clusters:
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X2)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg = silhouette_score(X2, cluster_labels)
    print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")


# In[109]:


model4 = KMeans(n_clusters=5, random_state=42)
visualizer = SilhouetteVisualizer(model4, size=(1080, 720))

visualizer.fit(X2)    # Fit the data to the visualizer
visualizer.poof()


# #### *iii. Apply K Means*

# In[110]:


model = KMeans(n_clusters=5, random_state=42)
#model.fit(X2)??


# In[111]:


y_kmeans = model.fit_predict(X2)


# In[112]:


X2.head()


# In[113]:


df['clusterx2_k'] = y_kmeans
df.head()


# In[114]:


df['clusterx2_k'].unique()


# In[ ]:





# #### *iv. Visualizing and Labeling All the Clusters*

# In[115]:


centers = model.cluster_centers_ # merkez noktasi koordinatlari
centers


# In[116]:


plt.figure(figsize=(15, 9))
plt.scatter(df["Annual_Income"], df["Spending_Score"], c = df.clusterx2_k, cmap = "viridis", alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('Customer segments')
plt.xlabel('Annual income of customer')
plt.ylabel('Annual spend from customer on site');


# ### Hierarchical Clustering

# ### *i. Determine optimal number of clusters using Dendogram*

# ### Clustering based on Age and Spending Score- x1

# In[117]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[118]:


X1.head()


# In[119]:


hc_ward = linkage(y=X1, method="ward")
hc_complete = linkage(X1, "complete")
hc_average = linkage(X1, "average")
hc_single = linkage(X1, "single")


# In[120]:


plt.figure(figsize=(20, 12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)
plt.show()


# In[121]:


plt.figure(figsize = (12,5))
dendrogram(hc_ward, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)
plt.show()


# In[ ]:





# ## Clustering based on Annual Income and Spending Score- x2

# In[131]:


X2.head()


# In[132]:


hc_ward = linkage(y=X2, method="ward")
hc_complete = linkage(X2, "complete")
hc_average = linkage(X2, "average")
hc_single = linkage(X2, "single")


# In[133]:


plt.figure(figsize=(20, 12))

plt.subplot(221)
plt.title("Ward")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_ward, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)

plt.subplot(222)
plt.title("Complete")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_complete, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)

plt.subplot(223)
plt.title("Average")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_average, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)

plt.subplot(224)
plt.title("Single")
plt.xlabel("Observations")
plt.ylabel("Distance")
dendrogram(hc_single, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)
plt.show()


# In[134]:


plt.figure(figsize = (12,5))
dendrogram(hc_ward, truncate_mode="lastp", p=10,
           show_contracted=True, leaf_font_size=10)
plt.show()


# ### ii. *Apply Agglomerative Clustering*

# #### Age and Spending Score- x1

# In[122]:


from sklearn.cluster import AgglomerativeClustering


# In[123]:


from sklearn.metrics import silhouette_score


# In[124]:


K = range(2,11)

for k in K:
    model = AgglomerativeClustering(n_clusters = k)
    model.fit_predict(X1)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(X1, model.labels_)}')


# In[125]:


model = AgglomerativeClustering(n_clusters=4, affinity = "euclidean", linkage = "ward")
clusters = model.fit_predict(X1)


# In[126]:


df["clusterx1_h"] = clusters
df.head()


# In[127]:


df.clusterx1_h.unique()


# In[128]:


plt.figure(figsize = (20,10))
sns.scatterplot(x='Age', y='Spending_Score', hue = df.clusterx1_h, data=df ,palette="bright")
plt.show()


# In[ ]:





# #### Annual Income and Spending Score- x2

# In[135]:


from sklearn.cluster import AgglomerativeClustering


# In[136]:


from sklearn.metrics import silhouette_score


# In[137]:


K = range(2,11)

for k in K:
    model = AgglomerativeClustering(n_clusters = k)
    model.fit_predict(X2)
    print(f'Silhouette Score for {k} clusters: {silhouette_score(X2, model.labels_)}')


# In[138]:


model = AgglomerativeClustering(n_clusters=5, affinity = "euclidean", linkage = "ward")
clusters = model.fit_predict(X2)


# In[139]:


df["clusterx2_h"] = clusters
df.head()


# In[140]:


plt.figure(figsize=(20, 10))
sns.scatterplot(x='Annual_Income', y='Spending_Score',
                hue=df.clusterx2_h, data=df, palette="bright")
plt.show()


# In[141]:





# ### iii. *Visualizing and Labeling All the Clusters* 

# #### Age and Spending Score- x1

# In[ ]:





# #### Annual Income and Spending Score- x2

# In[142]:


plt.figure(figsize=(20, 10))

plt.subplot(121)
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='clusterx2_k', data=df, palette=['green', 'orange',
                                                                                  'brown', 'dodgerblue', 'red'])
plt.title("K_means_X2")

plt.subplot(122)
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='clusterx2_h', data=df, palette=['orange', 'green',
                                                                                  'red', 'dodgerblue', 'brown'])
plt.title("Agg_X2")
plt.show()


# #### Interpretation based on Age and Spending Score- x1

# In[160]:


df['clusterx1_k'] = y_kmeans
df.head()


# In[161]:


plt.title("clusters with the number of customers")
plt.xlabel("clusters")
plt.ylabel("Count")
ax = df.clusterx1_k.value_counts().plot(kind='bar')
ax.bar_label(ax.containers[0])


# In[150]:


plt.figure(figsize=(14, 10))
plt.title("Men VS Women ratio in each cluster")
plt.ylabel("Count")
ax = sns.countplot(x=df.clusterx1_k, hue=df.Gender)
for p in ax.containers:
    ax.bar_label(p)
plt.show()


# In[153]:


df["clusterx2_h"] = clusters


# In[154]:


plt.title("Men VS Women ratio in each cluster")
plt.ylabel("Count")
ax = sns.countplot(x=df.clusterx2_h, hue=df.Gender)
for p in ax.containers:
    ax.bar_label(p)


# In[ ]:





# In[ ]:





# ### Conclusion

# **cluster 0** : The average age is around 55, both annula_income and spending_scores are on average. 
# It should be researched what can be done to direct to more spending.
# 
# **cluster 1**: The average age is around 45, the annula_income is high but the spending_scores are very low. 
# This group is our target audience and specific strategies should be developed to drive this group to spend.
# 
# **cluster 2** :The average age is around 30. The annula_income is high and spending_scores are very high. 
# This group consists of our loyal customers. Our company derives the main profit from this group. Very 
# special promotions can be made in order not to miss it.    
#     
# **cluster 3**: The average age is around 25.both annula_income and spending_scores are on average. 
# It should be researched what can be done to direct to more spending.

# #### Interpretation based on Annual Income and Spending Score- x2

# In[162]:


df.clusterx2_k.value_counts()


# In[163]:


plt.figure(figsize=(14, 10))

ax = sns.countplot(data=df, x='clusterx2_k')
for p in ax.containers:
    ax.bar_label(p)
plt.show()


# In[164]:


plt.figure(figsize=(14, 10))
plt.title("Men VS Women ratio in each cluster")
plt.ylabel("Count")
ax = sns.countplot(x=df.clusterx2_k, hue=df.Gender)
for p in ax.containers:
    ax.bar_label(p)
plt.show()


# In[166]:


df.groupby(["Gender", "clusterx2_k"]).mean()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Conclusion

# ### Female
# 
# **cluster 0** : The average age is around 40, both annula_income and spending_scores are on average. 
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 45, the annula_income is very high but the spending_scores is low.
# This group is our target audience and special strategies need to be developed for this group.    
# 
# **cluster 2** :The average age is around 45. Both annula_income and spending_scores are low. It can be 
# directed to shopping with gift certificates.
# 
# **cluster 3**: The average age is around 25. Low annual_incomes but very high spending scores. This 
# group does a lot of shopping, but they do not bring much profit.
# 
# **cluster 4**: The average age is around 30, the annual income and the spending_score
# is very high. This group consists of our loyal customers. Our company derives the main profit from this group. 
# Very special promotions can be made in order not to miss it.

# ### Male
# 
# **cluster 0** : The average age is around 45, both annula_income and spending_scores are on average. 
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 40, the annula_income is very high but the spending_scores is very low.
# This group is our target audience and special strategies need to be developed for this group.    
# 
# **cluster 2** :The average age is around 50. Both annula_income and spending_scores are low. It can be 
# directed to shopping with gift certificates.
# 
# **cluster 3**: The average age is around 25. Low annual_incomes but very high spending scores. This 
# group does a lot of shopping, but they do not bring much profit.
# 
# **cluster 4**: The average age is around 30, the annual income and the spending_score
# is very high. This group consists of our loyal customers. Our company derives the main profit from this group. 
# Very special promotions can be made in order not to miss it.

# In[ ]:





# **cluster 0** : The average age is around 40, both annula_income and spending_scores are on average. 
# It should be researched what can be done to direct more spending.
# 
# **cluster 1**: The average age is around 30, both annula_income and spending_scores are very high. 
# This group consists of our loyal customers. Our company derives the main profit from this group. Very 
# special promotions can be made in order not to miss it.
# 
# **cluster 2** :The average age is around 45. Both annula_income and spending_scores are low. It can be 
# directed to shopping with gift certificates.
# 
# **cluster 3**: The average age is around 25. Low annual_incomes but very high spending scores. This 
# group does a lot of shopping, but they do not bring much profit.
# 
# **cluster 4**: The average age is around 40, their annual income is very high but their spending_score
# is very low. This group is our target audience and special strategies need to be developed for this 
# group.

# 
# 

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
