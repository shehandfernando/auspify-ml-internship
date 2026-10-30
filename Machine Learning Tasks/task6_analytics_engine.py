import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Dataset.csv')

# ==========================================
# Step 1: Advanced Feature Engineering
# ==========================================
# We will predict whether a title is a TV Show (1) or Movie (0) using 
# complex categorical features (rating and country) via One-Hot Encoding.
df = df.dropna(subset=['type', 'rating', 'country'])

X = df[['rating', 'country']] 
y = (df['type'] == 'TV Show').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an automated preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['rating', 'country'])
    ])

# ==========================================
# Step 2 & 3: Build and compare multiple models
# ==========================================
# Pipeline A: Random Forest Classifier
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# Pipeline B: Logistic Regression
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

pipelines = {'Random Forest': rf_pipeline, 'Logistic Regression': lr_pipeline}
results = {}

print("Training ML pipelines and comparing performance...\n")
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"[{name}] Accuracy: {acc:.4f}")

# ==========================================
# Step 4: Generate automated insights & MLOps
# ==========================================
best_model_name = max(results, key=results.get)
best_model = pipelines[best_model_name]

print(f"\n[Automated Business Insight]")
print(f"The best performing classification engine is {best_model_name}.")

# MLOps Fundamental: Save the trained pipeline to disk for future deployment
model_filename = 'production_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Model successfully packaged and saved as '{model_filename}' for production.")

# ==========================================
# Step 5: Present findings through visual reports
# ==========================================
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='mako')
plt.title('ML Pipeline Performance Comparison')
plt.ylabel('Prediction Accuracy')
plt.ylim(0, 1.0)

# Add text labels on top of the bars
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    
plt.savefig('pipeline_comparison_report.png')
print("\nVisual report saved as 'pipeline_comparison_report.png'")