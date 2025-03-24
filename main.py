This script covers the basic steps for a personal fitness tracker. 
You can extend this script by adding more features, improving the predictive model, and integrating it with real fitness data sources 
(e.g., fitness trackers, mobile apps, etc.).

Python script that includes the following steps:

Data Collection: Collect fitness data (e.g., steps, distance, calories, etc.).
Data Preprocessing: Clean and preprocess the collected data.
Data Analysis: Analyze the data to extract insights.
Predictive Modeling: Use scikit-learn to build a predictive model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Data Collection
# For simplicity, we'll create a sample dataset
data = {
    'Date': pd.date_range(start='2025-01-01', periods=100, freq='D'),
    'Steps': [i + (i % 10) * 10 for i in range(100)],
    'Distance_km': [i * 0.05 for i in range(100)],
    'Calories': [i * 0.04 for i in range(100)]
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Normalize the data
df['Steps'] = (df['Steps'] - df['Steps'].mean()) / df['Steps'].std()
df['Distance_km'] = (df['Distance_km'] - df['Distance_km'].mean()) / df['Distance_km'].std()
df['Calories'] = (df['Calories'] - df['Calories'].mean()) / df['Calories'].std()

# Step 3: Data Analysis
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Steps'], label='Steps')
plt.plot(df['Date'], df['Distance_km'], label='Distance (km)')
plt.plot(df['Date'], df['Calories'], label='Calories')
plt.xlabel('Date')
plt.ylabel('Normalized Values')
plt.legend()
plt.title('Fitness Data Over Time')
plt.show()

# Step 4: Predictive Modeling
# We'll predict Calories burned based on Steps and Distance
X = df[['Steps', 'Distance_km']]
y = df['Calories']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calories')
plt.show()
