import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

attrition = pd.read_csv('attrition.csv')

print(attrition.head(8))

constant_features = []
for col in attrition.columns:
    if attrition[col].nunique() == 1:
        constant_features.append(col)

monotonic_features = []
for col in attrition.select_dtypes(include=['number']):
    if attrition[col].is_monotonic_increasing or attrition[col].is_monotonic_decreasing:
        monotonic_features.append(col)

features_to_remove = constant_features + monotonic_features
attrition_new = attrition.drop(columns=features_to_remove)

print(attrition_new)

# Task 3: Find and handle missing values

print(attrition_new.isnull().sum())

# Remove column is a risk of losing potentially valuable information.
# It can be interesting if we consider there are too much data missing, as we can see in the BusinessTravel column.
# Imputation with the median provides a reliable central value without distorting the results which can be very different.

threshold = len(attrition_new) * 0.5
attrition_new = attrition_new.dropna(thresh=threshold, axis=1)

missing_values = attrition_new.isnull().sum()

for col in missing_values[missing_values > 0].index:
    if attrition_new[col].dtype == 'object':
        attrition_new[col].fillna(attrition_new[col].mode()[0], inplace=True)
    else:
        attrition_new[col].fillna(attrition_new[col].median(), inplace=True)

attrition_imputation = attrition_new

print(attrition_imputation.to_string())

# Task 4: Transform categorical features into numerical ones
attrition_numerical = pd.get_dummies(attrition_imputation, drop_first=True).astype(int)
print(attrition_numerical)

# Task 5: Display stats for numerical features and frequencies for categorical features
numerical_features = attrition_imputation.select_dtypes(include=['number']).columns
categorical_features = attrition_imputation.select_dtypes(include=['object']).columns

print("\nNumerical features, mean, median, minimum, and maximum:")
for col in numerical_features:
    print(f"{col}: Mean={attrition_imputation[col].mean()}, Median={attrition_imputation[col].median()}, Min={attrition_imputation[col].min()}, Max={attrition_imputation[col].max()}")

print("\nCategorical feature frequency:")
for col in categorical_features:
    print(f"\n{attrition_imputation[col].value_counts()}")

# Task 6: Normalize numerical features
attrition_normalized = (attrition_numerical - attrition_numerical.min()) / (attrition_numerical.max() - attrition_numerical.min())
print(attrition_normalized)

# Task 7: Composite plot with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of a categorical feature (e.g., "Attrition")
sns.histplot(attrition_imputation, x="Attrition", ax=axs[0, 0])
axs[0, 0].set_title("Histogram of Attrition")

# Pie chart of the target variable ("Attrition")
attrition_imputation['Gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=axs[0, 1], startangle=90)
axs[0, 1].set_title("Pie chart of Gender")

# Box plot of a numerical feature (e.g., "Age")
sns.boxplot(data=attrition_imputation, x="Age", ax=axs[1, 0])
axs[1, 0].set_title("Box plot of Age")

# Scatter plot of two numerical features (e.g., "Age" vs. "MonthlyIncome") colored by Attrition
sns.scatterplot(data=attrition_imputation, x="Age", y="MonthlyIncome", hue="Attrition", ax=axs[1, 1])
axs[1, 1].set_title("Scatter plot of Age vs Monthly Income by Attrition")

# Display the plots
plt.tight_layout()
plt.show()