## Project Overview

**Project Title**: Customer-Segmentation Analysis. 
# dataset link: [https://www.kaggle.com/code/analystoleksandra/marketing-analytics-customer-segmentation/notebook?select=ifood_df.csv](https://www.kaggle.com/datasets/jackdaoud/marketing-data)


## Introduction
The goal of this data analytics project is to perform customer segmentation analysis for an e-commerce company. By analyzing customer behavior and purchase patterns, we aim to group customers into distinct segments. These segments can be used to improve marketing strategies, enhance customer satisfaction, and optimize business operations.

## Key Concepts and Challenges
- Data Collection: We use a dataset containing customer demographics, purchase history, and relevant transaction data.

- Data Exploration and Cleaning: Understanding the dataset structure and handling missing or inconsistent data.

- Descriptive Statistics: Computing key customer metrics such as income, purchase frequency, and total spending.

- Exploratory Data Analysis (EDA): Analyzing the distribution of key variables and detecting patterns before clustering.

- Customer Segmentation: Applying clustering techniques (K-Means) to categorize customers based on purchasing behaviors and demographics.

- Visualization: Using charts and graphs to illustrate customer segments and their characteristics.

- Insights and Recommendations: Analyzing the characteristics of each segment to provide actionable business insights.


## 1. Data Collection:
This projects starts by  Obtaining  a dataset containing customer information, purchase history, and relevant data

- **Data Information:**
The data contains 2,205 observations and 39 columns. The dataset description on the card does not match the actual columns in the dataset. The below list contains actual columns from the dataset and the assumed descriptions from the column's names.



### 2. Data preprocessinng and cleaning 
- import necessary libraries. 
- load data.
- preview data
- check data information (amount of columns, enteries, data types)
- check column statistics.
- Check for any null values in the dataset.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

```python
df = pd.read_csv('ifood_df.csv') ##load data 
df.head() ## view data 
df.describe() ## data summary\ statistics
df.info() ## data information
df.isnull().sum() ## check for nulll values
```
The data contains 2,205 observations and 39 columns. The data contains has no null values 


### 3. Data Analysis & Findings
- **Calculate  average purchase value.**
```python
average_purchase_value = df['MntTotal'].mean()
average_purchase_value
```

- **frequency of purchases.**
```python
purchase_frequency = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1).mean()
purchase_frequency
```
- **total spendimg on products** 
```python
total_spending_by_category = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()
total_spending_by_category
```
- **distinct frequent purchase of each product.**
```python
distinct_purchase_frequency = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].gt(0).sum()
distinct_purchase_frequency

distinct_purchase_frequency = {
    'MntWines': 2192,
    'MntFruits': 1812,
    'MntMeatProducts': 2204,
    'MntFishProducts': 1826,
    'MntSweetProducts': 1794,
    'MntGoldProds': 2144
}

# Plotting the distinct purchase frequency
plt.figure(figsize=(10, 6))
plt.bar(distinct_purchase_frequency.keys(), distinct_purchase_frequency.values(), color='teal')
plt.title('Distinct Purchase Frequency for Each Product')
plt.xlabel('Product Category')
plt.ylabel('Number of Purchases (Distinct)')
plt.xticks(rotation=45)
plt.show()

```
- **Calculate the total number of purchases for each customer (sum across all purchase channels).** 
```python
df['Total_Purchases'] = df[['NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']].sum(axis=1)

# Plot the distribution of total purchases
plt.figure(figsize=(12, 6))
plt.hist(df['Total_Purchases'], bins=15, color='skyblue', edgecolor='black')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.title('Distribution of Total Purchases per Customer')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```
- **Age Distribution**
```python 
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of Age
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
```

- **Distribution of Income**
```python
  
plt.figure(figsize=(6,4))
sns.histplot(df['Income'], bins=30, kde=True)
plt.title("Income Distribution")
plt.show()
```
- **Recency Analysis**
```python
plt.figure(figsize=(6,4))
sns.histplot(df['Recency'], bins=30, kde=True)
plt.title("Recency Distribution")
plt.show()
```


- **Total Spending by Age Group**
```python
# Define age groups
bins = [18, 25, 35, 45, 55, 65, 100]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Calculate total spending for each age group
age_group_spending = df.groupby('Age_Group')['MntTotal'].sum().reset_index()

# Plot spending by age group
plt.figure(figsize=(12, 6))
sns.barplot(data=age_group_spending, x='Age_Group', y='MntTotal', palette='Blues')
plt.title('Total Spending by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Spending')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

# Kmeans clustering
- **Correlation analysis:**
- Selecting relevant columns for each segmentation:
in this case we would be grouping the columns into segments and applying heatmap to see the correlations between them and to know which analysis is best we do. 

```python
# grouping relevant columns
rfm_cols = ["Recency", "MntTotal", "NumStorePurchases", "NumWebPurchases", "NumCatalogPurchases"] 
product_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
demographic_cols = ["Income", "Age", "Kidhome", "Teenhome"]
```

```python
# Combine all selected columns into one DataFrame
selected_cols = rfm_cols + product_cols + demographic_cols
df_selected = df[selected_cols]
```

```python
# Compute correlation matrices
correlation_rfm = df[rfm_cols].corr()
correlation_product = df[product_cols].corr()
correlation_demographic = df[demographic_cols].corr()
```


```python
# Plot heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(correlation_rfm, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=axes[0])
axes[0].set_title("Correlation Matrix - RFM Segmentation")

sns.heatmap(correlation_product, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=axes[1])
axes[1].set_title("Correlation Matrix - Product-Based Segmentation")

sns.heatmap(correlation_demographic, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=axes[2])
axes[2].set_title("Correlation Matrix - Demographic Segmentation")

plt.tight_layout()
plt.show()
```
```python
# Display top correlated features
correlation_unstacked = df_selected.corr().unstack().sort_values(ascending=False)
print("Top Correlations:")
print(correlation_unstacked[correlation_unstacked < 1].head(10))  # Exclude self-correlations
```

the rfm and product base segmentation have strong positive correlations between them. with the product base segment having higher positive correlations. 
we would dive moe into the analysis. 

# Apply kmeans clustering for rfm
```python 

# Selecting relevant columns for RFM segmentation
rfm_cols = ["Recency", "MntTotal", "NumStorePurchases", "NumWebPurchases", "NumCatalogPurchases"] 

df_rfm = df[rfm_cols]

# Compute correlation matrix
correlation_rfm = df_rfm.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_rfm, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix - RFM Segmentation")
plt.show()

# Display top correlated features
correlation_unstacked = df_rfm.corr().unstack().sort_values(ascending=False)
print("Top Correlations:")
print(correlation_unstacked[correlation_unstacked < 1].head(10))  # Exclude self-correlations

# Function for KMeans clustering, Silhouette & Elbow Method, and PCA visualization
def perform_kmeans_pca(df_subset, cols, segmentation_name):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_subset[cols])
    
    # Determine optimal clusters
    wcss = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(df_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, cluster_labels))
    
    # Plot Elbow Method
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal K')
    plt.show()
    
    # Plot Silhouette Score
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Analysis')
    plt.show()
    
    # Choose optimal clusters
    optimal_clusters = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal clusters for {segmentation_name}: {optimal_clusters}")
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    df_subset[f"{segmentation_name}_Cluster"] = kmeans.fit_predict(df_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    df_subset["PCA1"] = df_pca[:, 0]
    df_subset["PCA2"] = df_pca[:, 1]
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df_subset["PCA1"], y=df_subset["PCA2"], hue=df_subset[f"{segmentation_name}_Cluster"], palette="viridis", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"KMeans Clustering Visualization Using PCA - {segmentation_name}")
    plt.legend(title="Cluster")
    plt.show()
    
    # Boxplot analysis for MntTotal and clusters
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_subset[f"{segmentation_name}_Cluster"], y=df_subset["MntTotal"], palette="viridis")
    plt.xlabel("Cluster")
    plt.ylabel("MntTotal")
    plt.title(f"Boxplot of MntTotal by {segmentation_name} Clusters")
    plt.show()
    
    # Boxplot analysis for Recency and clusters
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_subset[f"{segmentation_name}_Cluster"], y=df_subset["Recency"], palette="coolwarm")
    plt.xlabel("Cluster")
    plt.ylabel("Recency")
    plt.title(f"Boxplot of Recency by {segmentation_name} Clusters")
    plt.show()
    
    return df_subset

# Perform KMeans, Silhouette & Elbow Method, and PCA on RFM segmentation
df_rfm_segmented = perform_kmeans_pca(df, rfm_cols, "RFM")
```
