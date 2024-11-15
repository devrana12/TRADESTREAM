import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from transformers import pipeline
import yfinance as yf

# stock data using yfinance
def get_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

#AMP (AYUSH-MAYANK-PRANAV Company)
stock_data = get_stock_data('AMP', '2024-05-01', '2024-10-01')

# Sample weather news
weather_news = [
    "Record harvests expected this year due to favorable weather conditions.",
    "Innovative farming techniques boost crop yields significantly.",
    "New trade agreements open up markets for American grain exports.",
    "Grain prices rise as demand increases in international markets.",
    "Farmers report bumper crop yields as technology improves production.",
    "Drought relief efforts lead to resurgence in grain production.",
    "Strong commodity prices boost profits for local farmers.",
    "Favorable growing conditions lead to record-breaking wheat harvest.",
    "Investments in sustainable agriculture pay off with higher yields.",
    "New seed varieties show promise in enhancing grain resilience."
]

# Sentiment Analysis using transformers pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment_scores(news):
    sentiment_scores = []
    for article in news:
        sentiment = sentiment_pipeline(article)[0]
        if sentiment['label'] == 'POSITIVE':
            sentiment_scores.append(sentiment['score'])  # Positive score
        else:
            sentiment_scores.append(-sentiment['score'])  # Negative score
    return sentiment_scores

# Get sentiment scores for weather news
sentiment_scores = get_sentiment_scores(weather_news)
print("Sentiment Scores: ", sentiment_scores)

# Prepare dataset (stock data and sentiment)
def prepare_dataset(stock_data, sentiment_scores):
    if len(sentiment_scores) < len(stock_data):
        sentiment_scores += [0] * (len(stock_data) - len(sentiment_scores))  # Padding with neutral sentiments
    
    stock_data['Sentiment'] = sentiment_scores[:len(stock_data)]
    stock_data['Price Change'] = stock_data['Close'].pct_change()  # change in price
    stock_data.dropna(inplace=True)
    return stock_data

# Prepare dataset combining stock data and sentiment
dataset = prepare_dataset(stock_data, sentiment_scores)

# Step 4: Features and target
X = dataset[['Sentiment']]  # Features
y = np.where(dataset['Price Change'] > 0, 1, 0)  # 1 for price increase, 0 for price decrease

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Model Accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Step 7: Virtual Money Simulation
def virtual_trade(stock_data, model, scaler, initial_money=1000):
    stock_data['Predicted Change'] = model.predict(scaler.transform(stock_data[['Sentiment']]))
    
    virtual_money = initial_money  # Starting virtual money
    for i, row in stock_data.iterrows():
        if row['Predicted Change'] == 1:  # If model predicts price increase
            # Increase virtual money based on actual price change
            if not np.isnan(row['Price Change']):  
                virtual_money *= (1 + row['Price Change'])
    
    return virtual_money

# Simulate virtual trading with the model
initial_money = 1000  
final_amount = virtual_trade(dataset, model, scaler, initial_money)
print(f'Deposited Money: INDIAN RUPEES {initial_money:.2f}')
print(f'Final Virtual Money after trading: ${final_amount:.2f}') 

