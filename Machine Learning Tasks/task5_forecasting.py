import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Dataset.csv')

# ==========================================
# Step 1: Prepare time-based features
# ==========================================
# Convert the 'date_added' string column to proper datetime objects
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), format='mixed', errors='coerce')

# Drop any rows where the date couldn't be parsed
df = df.dropna(subset=['date_added'])

# Extract the year the content was added
df['year_added'] = df['date_added'].dt.year

# ==========================================
# Step 2: Analyze historical release patterns
# ==========================================
# Count how many titles were added each year
yearly_counts = df.groupby('year_added').size().reset_index(name='count')

# Remove data before 2014 as Netflix streaming started scaling then,
# preventing early flatline years from skewing the recent trend.
yearly_counts = yearly_counts[yearly_counts['year_added'] >= 2014]

# ==========================================
# Step 3: Build forecasting models
# ==========================================
# We use Linear Regression to identify the upward/downward trend over time
X = yearly_counts[['year_added']]
y = yearly_counts['count']

model = LinearRegression()
model.fit(X, y)

# ==========================================
# Step 4: Generate future predictions
# ==========================================
# Predict content additions for the next few years
future_years = pd.DataFrame({'year_added': [2022, 2023, 2024, 2025]})
future_predictions = model.predict(future_years)

future_df = pd.DataFrame({
    'Year': future_years['year_added'],
    'Predicted_Count': future_predictions.astype(int) # Round to whole titles
})

print("Historical Content Additions (Recent):")
print(yearly_counts.tail())
print("\nFuture Trend Forecast:")
print(future_df.to_markdown(index=False))

# ==========================================
# Step 5: Evaluate forecasting accuracy
# ==========================================
# Calculate how well our linear trendline fits the historical data
y_pred_hist = model.predict(X)
r2 = r2_score(y, y_pred_hist)
print(f"\nModel R-Squared (Fit Accuracy): {r2:.4f}")

# Visualize the historical data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(X, y, marker='o', label='Historical Additions', color='blue')
plt.plot(future_years, future_predictions, marker='X', linestyle='--', color='red', label='Forecasted Additions')
plt.title('Netflix Content Addition Trend Forecast')
plt.xlabel('Year')
plt.ylabel('Number of Titles Added')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('trend_forecast.png')
print("\nVisualization saved as 'trend_forecast.png'")