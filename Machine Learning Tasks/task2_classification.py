import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('Dataset.csv')

# ==========================================
# Step 1: Select relevant dataset features
# ==========================================
# 'type' is our target variable (what we want to predict).
# We'll use 'director', 'country', 'rating', and 'listed_in' as features.
features = ['director', 'country', 'rating', 'listed_in']

# Handle any potential missing values by replacing them with 'Unknown'
df[features] = df[features].fillna('Unknown')

# ==========================================
# Step 2: Encode categorical variables
# ==========================================
# Machine learning models need numbers, not text.
# LabelEncoder converts unique strings into unique integers.
X = pd.DataFrame()
for feature in features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(df[feature].astype(str))

# Encode the target variable ('Movie' -> 0, 'TV Show' -> 1)
target_le = LabelEncoder()
y = target_le.fit_transform(df['type'])

# ==========================================
# Step 3: Train classification models
# ==========================================
# Split the data: 80% for training the model, 20% for testing it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ==========================================
# Step 4: Evaluate prediction performance
# ==========================================
# Make predictions on the unseen test data
y_pred = rf_model.predict(X_test)

# ==========================================
# Step 5: Compare model accuracy
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_le.classes_)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)