# IRIS FLOWER CLASSIFICATION PROJECT IN COLAB

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load the IRIS dataset (no need to upload file, we use sklearn's built-in dataset)
from sklearn.datasets import load_iris
iris = load_iris()

# Step 3: Convert to pandas DataFrame for better handling
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

# Step 4: Display first few rows
print("🔹 First 5 rows of the dataset:")
print(df.head())

# Step 5: Dataset information
print("\n🔹 Dataset Information:")
print(df.info())

# Step 6: Check for missing values
print("\n🔹 Checking for missing values:")
print(df.isnull().sum())

# Step 7: Basic statistical summary
print("\n🔹 Statistical Summary:")
print(df.describe())

# Step 8: Visualize data
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle("🔍 Pairplot of IRIS Dataset", y=1.02)
plt.show()

# Step 9: Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='Blues')
plt.title("📊 Feature Correlation Heatmap")
plt.show()

# Step 10: Prepare data for model
X = df.drop('species', axis=1)
y = df['species']

# Step 11: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 12: Train the model using K-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 13: Make predictions
y_pred = model.predict(X_test)

# Step 14: Evaluate the model
print("\n✅ Model Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
