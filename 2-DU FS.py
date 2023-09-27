
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Clients-data.csv')

# Select a random sample

random_sample = df.sample(n=10000)

# Separate the target variable 'EthMPAO' and the features
X = random_sample.drop('EthMPAO', axis=1)
y = random_sample['EthMPAO']


# Convert categorical features to numerical using one-hot encoding
X_encoded = pd.get_dummies(X)
X_encoded.dropna(axis=0, inplace=True)
y = y.loc[X_encoded.index]  # Update y to match the filtered X_encoded

clf = RandomForestClassifier(random_state=42)
clf.fit(X_encoded, y)


# Initialize a random forest classifier
clf = RandomForestClassifier(random_state=42)

# Fit the model to the data
clf.fit(X_encoded, y)

# Get feature importances from the trained model
feature_importances = clf.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the top N most important features
top_n = 5  # We can change this value to show more or fewer top features
plt.figure(figsize=(12, 6))

sns.barplot(data=feature_importance_df.head(top_n), x='Importance', y='Feature', palette='viridis')

plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Feature Importance for EthMPAO Classification')

plt.tight_layout()
plt.show()
