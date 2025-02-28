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






















## Findings

- **Customer Demographics**: The dataset includes customers from various age groups, with sales distributed across different categories such as Clothing and Beauty.
- **High-Value Transactions**: Several transactions had a total sale amount greater than 1000, indicating premium purchases.
- **Sales Trends**: Monthly analysis shows variations in sales, helping identify peak seasons.
- **Customer Insights**: The analysis identifies the top-spending customers and the most popular product categories.

## Reports

- **Sales Summary**: A detailed report summarizing total sales, customer demographics, and category performance.
- **Trend Analysis**: Insights into sales trends across different months and shifts.
- **Customer Insights**: Reports on top customers and unique customer counts per category.

## Conclusion

This project serves as a comprehensive introduction to SQL for data analysts, covering database setup, data cleaning, exploratory data analysis, and business-driven SQL queries. The findings from this project can help drive business decisions by understanding sales patterns, customer behavior, and product performance.

## How to Use

1. **Clone the Repository**: Clone this project repository from GitHub.
2. **Set Up the Database**: Run the SQL scripts provided in the `database_setup.sql` file to create and populate the database.
3. **Run the Queries**: Use the SQL queries provided in the `analysis_queries.sql` file to perform your analysis.
4. **Explore and Modify**: Feel free to modify the queries to explore different aspects of the dataset or answer additional business questions.

## Author - Zero Analyst

This project is part of my portfolio, showcasing the SQL skills essential for data analyst roles. If you have any questions, feedback, or would like to collaborate, feel free to get in touch!

### Stay Updated and Join the Community

For more content on SQL, data analysis, and other data-related topics, make sure to follow me on social media and join our community:

- **YouTube**: [Subscribe to my channel for tutorials and insights](https://www.youtube.com/@zero_analyst)
- **Instagram**: [Follow me for daily tips and updates](https://www.instagram.com/zero_analyst/)
- **LinkedIn**: [Connect with me professionally](https://www.linkedin.com/in/najirr)
- **Discord**: [Join our community to learn and grow together](https://discord.gg/36h5f2Z5PK)

Thank you for your support, and I look forward to connecting with you!
