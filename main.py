import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

# Load the data
diabete = pd.read_csv('diabetes.csv')

print(diabete.head(8))

# Removing constant and monotonic features
constant_features = []
for col in diabete.columns:
    if diabete[col].nunique() == 1:
        constant_features.append(col)

monotonic_features = []
for col in diabete.select_dtypes(include=['number']):
    if diabete[col].is_monotonic_increasing or diabete[col].is_monotonic_decreasing:
        monotonic_features.append(col)

features_to_remove = constant_features + monotonic_features
diabete = diabete.drop(columns=features_to_remove)

print(diabete.head(8))

# Check for missing values
print(diabete.isnull().sum())

# Imputation with the median (There is no missing data so imputation is not really important)
threshold = len(diabete) * 0.5
diabete = diabete.dropna(thresh=threshold, axis=1)

missing_values = diabete.isnull().sum()

for col in missing_values[missing_values > 0].index:
    if diabete[col].dtype == 'object':
        diabete[col].fillna(diabete[col].mode()[0], inplace=True)
    else:
        diabete[col].fillna(diabete[col].median(), inplace=True)

diabete = pd.get_dummies(diabete, drop_first=True).astype(int)
print(diabete.head(8))

# Display stats for numerical features
print("\nNumerical features, mean, median, minimum, and maximum:")
for col in diabete:
    print(f"{col}: Mean={diabete[col].mean()}, Median={diabete[col].median()}, Min={diabete[col].min()}, Max={diabete[col].max()}")

# Normalize numerical features (All features are numerical)

diabete = (diabete - diabete.min()) / (diabete.max() - diabete.min())

# Plot correlation matrix
print("\nCorrelation Matrix:")
print(diabete.corr())

plt.figure(figsize=(10, 8))
sns.heatmap(diabete.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatterplot matrix of features
sns.pairplot(diabete, hue='Outcome', diag_kind='kde')

# Box and whisker plots
features = ['DiabetesPedigreeFunction', 'Age', 'SkinThickness', 'Insulin', 'BMI', 'Pregnancies', 'Glucose', 'BloodPressure']
plt.figure(figsize=(10, 5))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Outcome', y=feature, data=diabete, hue='Outcome', palette='Set2', dodge=False)
    plt.title(f"{feature}", fontsize=10)
    plt.ylabel(feature)

# Principal component analysis
plt.figure(figsize=(8, 6))
sns.scatterplot(data=diabete, x='Glucose', y='BMI', hue='Outcome', palette='coolwarm', alpha=0.7)
plt.title("Glucose vs BMI by Diabetes Outcome")
plt.xlabel("Glucose Level")
plt.ylabel("BMI")
plt.show()

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# k-Means clustering for k=2, 3, 4
k_values = [2, 3, 4]
fig, axes = plt.subplots(1, len(k_values), figsize=(18, 6))

for idx, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[idx].scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    axes[idx].set_title(f'k = {k}')

plt.suptitle("k-Means Clustering with k=2, 3, 4", fontsize=16)
plt.tight_layout()
plt.show()

# Results of rpart_importance function in R (Feature Importance)
features = ['Glucose', 'BMI', 'Age', 'Insulin', 'Skin', 'DPF', 'BP', 'Preg']
importance = [100.00, 59.98, 56.04, 42.80, 29.91, 0.00, 0.00, 0.00]

plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=features, hue=features, dodge=False, palette="coolwarm", legend=False)
plt.title("Feature Importance (rpart_importance function in R)", fontsize=16)
plt.xlabel("Importance (%)")
plt.ylabel("Features")
plt.show()

X = diabete.drop('Outcome', axis=1)  # Features
y = diabete['Outcome']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

models = {'Decision Tree': dt_model, 'Random Forest': rf_model, 'Naive Bayes': nb_model}
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC: {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

feature_importances = rf_model.feature_importances_
sns.barplot(x=feature_importances, y=X.columns)
plt.title("Feature Importances")
plt.show()
