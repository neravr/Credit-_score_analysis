import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('credit_score_cleaned_train.csv')

# Display the first few rows
print(data.head())

# Get the shape of the dataset
print(f"Dataset has {data.shape[0]} rows and {data.shape[1]} columns.")

# Summary of the dataset
print(data.info())

# Statistical summary of numerical columns
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Check column data types
print(data.dtypes)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Handling missing data
# Fill missing numerical values with median
data['age'] = data['age'].fillna(data['age'].median())
data['annual_income'] = data['annual_income'].fillna(data['annual_income'].mean())

# Fill missing categorical values with mode
data['occupation'] = data['occupation'].fillna(data['occupation'].mode()[0])


desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Example of correlation
# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
correlation_matrix = numeric_data.corr()

# Display correlation matrix
print("Correlation Matrix:\n", correlation_matrix)

##IQR
# Visualize numerical columns with boxplots before handling outliers
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 6))  # Reduced figure size
for i, col in enumerate(numerical_cols):
    plt.subplot(len(numerical_cols) // 3 + 1, 3, i + 1)  # Adjusted layout to 3 plots per row
    sns.boxplot(y=data[col], color='lightblue')
    plt.title(f"Before: {col}")
plt.tight_layout()
plt.show()

# Handle outliers using IQR and print number of outliers
outliers_count_dict = {}

for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

    # Store the count of outliers
    outliers_count_dict[col] = len(outliers)

    # Cap outliers to the lower and upper bounds
    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

# Print the number of outliers for each column
for col, outliers_count in outliers_count_dict.items():
    if outliers_count > 0:
        print(f"Number of outliers in {col}: {outliers_count}")

# Visualize numerical columns with boxplots after handling outliers
#plt.figure(figsize=(10, 6))  # Reduced figure size
#   plt.subplot(len(numerical_cols) // 3 + 1, 3, i + 1)  # Adjusted layout to 3 plots per row
 #   sns.boxplot(y=data[col], color='lightgreen')
  #  plt.title(col)  # Optional: Add a title to each subplot
#plt.tight_layout()  # Adjust layout to avoid overlapping plots
#plt.show()



# Slicing the dataframe
# Extract data for a specific month
january_data = data[data['month'] == 'January']
print("January Data:\n", january_data.head())
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Converting a column
# Converting `credit_score` to categorical labels
def credit_score_category(score):
    if score <= 3:
        return 'Poor'
    elif 4 <= score <= 6:
        return 'Average'
    else:
        return 'Good'
data['credit_score_category'] = data['credit_score'].apply(credit_score_category)

# GroupBy
grouped_data = data.groupby('occupation')['annual_income'].mean()
print("Average Annual Income by Occupation:\n", grouped_data)

# Merging two dataframes
# Create a dummy dataframe for merging
additional_data = pd.DataFrame({
    'customer_id': ['CUS_0xd40', 'CUS_0xab12'],
    'additional_info': ['Info1', 'Info2']})

merged_data = pd.merge(data, additional_data, on='customer_id', how='left')
print("Merged Data:\n", merged_data.head())


print("Data Cleaning Complete. Missing Values Handled.")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


# Cluster Analysis (KMeans)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Select relevant features for clustering
clustering_data = data[['annual_income', 'outstanding_debt', 'monthly_balance']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Elbow method to find the optimal number of clusters
inertia = []
k_range = range(1, 11)  # Checking k values from 1 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Print inertia values for each k
print("Inertia values for each number of clusters:")
for k, value in zip(k_range, inertia):
    print(f"k = {k}: Inertia = {value}")

# Plotting the elbow method graph
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.show()

# From the elbow plot, choose an optimal k (e.g., 4) and apply K-Means clustering
optimal_k = 4  # This value should be based on your elbow method result
print(f"\nApplying K-Means with optimal k = {optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_data)

# Print the cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Print the first few rows with the cluster assignments
print("\nFirst few rows with cluster assignments:")
print(data[['annual_income', 'outstanding_debt', 'monthly_balance', 'cluster']].head())

# Visualize clusters
sns.scatterplot(x='annual_income', y='outstanding_debt', hue='cluster', data=data, palette='viridis')
plt.title("Customer Clusters")
plt.show()


#Classifocation | Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Verify the column names to ensure 'credit_score_category' exists
print("Column names in dataset:", data.columns)

# Define features and target variable
X = data[['age', 'annual_income', 'occupation']]  # Adjust with relevant features
y = data['credit_score']  # Categorical target

# Convert categorical data (e.g., occupation) into numerical values
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Classification Accuracy: {accuracy}')


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define features and target variable
X = data[['age', 'occupation']]  # Adjust with relevant features
y = data['annual_income']

# Check for missing values
print("Missing values in features (X):")
print(X.isnull().sum())
print("\nMissing values in target (y):")
print(y.isnull().sum())

# Convert categorical data (e.g., occupation) into numerical values
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#6 Dimentionality Reduction
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.select_dtypes(include=['int64', 'float64']))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
data['cluster'] = kmeans.fit_predict(scaled_data)

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Add PCA results to the data
data['pca1'] = pca_result[:, 0]
data['pca2'] = pca_result[:, 1]

# Print PCA explained variance ratio
print("Explained Variance Ratio by PCA Components:")
print(pca.explained_variance_ratio_)

# Print KMeans cluster information
print("\nKMeans Cluster Centers (Centroids):")
print(kmeans.cluster_centers_)

# Print the number of samples in each cluster
cluster_counts = data['cluster'].value_counts()
print("\nNumber of samples in each cluster:")
print(cluster_counts)

# Visualize PCA results with clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca1', y='pca2', data=data, hue='cluster', palette='viridis', style='cluster', s=100)
plt.title('PCA of Data with KMeans Clusters')
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score



from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Ensure all outputs are displayed
pd.set_option('display.max_columns', None)

# Classification Analysis: Random Forest Classifier
print("Starting Classification Analysis...\n")
X_class = data[['age', 'annual_income', 'occupation']]
y_class = data['credit_score_category']

# Encode categorical variables
X_class = pd.get_dummies(X_class, drop_first=True)
le = LabelEncoder()
y_class = le.fit_transform(y_class)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_class = rf_model.predict(X_test)

# Print Classification Results
print("\nClassification Accuracy:", accuracy_score(y_test, y_pred_class))
print("\nClassification Report:\n", classification_report(y_test, y_pred_class))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Regression Analysis: Random Forest Regressor
print("\nStarting Regression Analysis...\n")
X_reg = data[['age', 'annual_income', 'occupation']]
y_reg = data['credit_score']
X_reg = pd.get_dummies(X_reg, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred_reg = rf_regressor.predict(X_test)

# Print Regression Results
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred_reg))
print("R-Squared:", r2_score(y_test, y_pred_reg))

# Plot Feature Importance for Regression
importances = rf_regressor.feature_importances_
feature_names = X_reg.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette="coolwarm")
plt.title("Feature Importance for Regression")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Behavioral Analysis: Correlation and Payment Behavior Analysis
print("\nStarting Behavioral Analysis...\n")
correlation_data = data[['age', 'annual_income', 'outstanding_debt', 'delay_from_due_date', 'credit_utilization_ratio']]
correlation_matrix = correlation_data.corr()

# Print Correlation Matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Payment Behavior and Delayed Payments
payment_delay_pattern = data.groupby('payment_behaviour')['delay_from_due_date'].mean().reset_index()
payment_delay_pattern = payment_delay_pattern.sort_values('delay_from_due_date', ascending=False)

# Print Payment Behavior Analysis
print("\nAverage Delay from Due Date by Payment Behaviour:")
print(payment_delay_pattern)

# Visualize Delayed Payments by Behavior
plt.figure(figsize=(10, 6))
sns.barplot(x='payment_behaviour', y='delay_from_due_date', data=payment_delay_pattern, palette='viridis')
plt.title('Average Delay from Due Date by Payment Behaviour')
plt.xlabel('Payment Behaviour')
plt.ylabel('Average Delay from Due Date (Days)')
plt.xticks(rotation=45)
plt.show()

# Time-Series Forecasting: Credit Score
print("\nStarting Time-Series Forecasting...\n")
data['month'] = pd.to_datetime(data['month'], format='%B')
data.set_index('month', inplace=True)

credit_score_series = data['credit_score']

# Split into train and test
train_size = int(len(credit_score_series) * 0.7)
train, test = credit_score_series[:train_size], credit_score_series[train_size:]

# ARIMA Model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate and Print Forecasting Results
mse = mean_squared_error(test, forecast)
print("Mean Squared Error for ARIMA Model:", mse)

# Visualize Forecasting Results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.title('Credit Score Forecasting using ARIMA')
plt.xlabel('Month')
plt.ylabel('Credit Score')
plt.legend()
plt.show()
