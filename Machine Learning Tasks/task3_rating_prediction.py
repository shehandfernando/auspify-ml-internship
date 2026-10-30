import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Ignore warnings about zero division for small rating categories
warnings.filterwarnings('ignore') 

# Load the dataset
df = pd.read_csv('Dataset.csv')

# ==========================================
# Step 1: Analyze rating categories
# ==========================================
# We'll use 'type', 'director', 'country', and 'listed_in' as predictive features
features = ['type', 'director', 'country', 'listed_in']

# Clean up missing data
df[features] = df[features].fillna('Unknown')
df = df.dropna(subset=['rating']) # Drop rows where the target rating itself is missing

# ==========================================
# Step 2: Prepare training datasets
# ==========================================
X = pd.DataFrame()
for feature in features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(df[feature].astype(str))

target_le = LabelEncoder()
y = target_le.fit_transform(df['rating'])

# ==========================================
# Step 3: Train classification models
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Random Forest to predict multi-class audience ratings
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# ==========================================
# Step 4: Optimize model performance
# Step 5: Evaluate prediction accuracy
# ==========================================
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Rating Prediction Accuracy: {accuracy:.4f}")
print("\nClassification Report:")

# Fix the ValueError by identifying which rating classes actually ended up in the test split
unique_labels = np.unique(np.concatenate((y_test, y_pred)))
target_names_filtered = [target_le.classes_[i] for i in unique_labels]

print(classification_report(y_test, y_pred, 
                            labels=unique_labels, 
                            target_names=target_names_filtered, 
                            zero_division=0))