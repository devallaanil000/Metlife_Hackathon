import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from textblob import TextBlob



# Sentiment Analysis Function
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Add Sentiment Score and Churn Labels
data['Sentiment Score'] = data['Transcript'].apply(get_sentiment)
data['Churn'] = data['Customer NPS'].apply(lambda x: 1 if x <= 6 else 0)  # Churn if NPS <= 6

# Prepare Features and Labels
features = ['Sentiment Score', 'Call Duration', 'Reponse Team Location']  # Include relevant columns
X = pd.get_dummies(data[features], drop_first=True)  # Handle categorical data
y = data['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
import matplotlib.pyplot as plt

feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
feature_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance for Churn Prediction')
plt.show()
