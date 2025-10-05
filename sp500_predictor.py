#!/usr/bin/env python3
"""
S&P 500 Price Prediction Model
Educational purposes only - NOT for actual trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SP500Predictor:
    def __init__(self):
        self.model = None
        self.features = []

    def download_data(self, start_date='2018-01-01', end_date=None):
        """Download S&P 500 historical data"""
        print("Downloading S&P 500 data...")

        # If no end date specified, use today
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        self.data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        # Flatten column names
        self.data.columns = [col[0] if isinstance(col, tuple) else col for col in self.data.columns]

        print(f"✓ Downloaded {len(self.data)} trading days")
        print(f"✓ Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")

        return self.data

    def engineer_features(self):
        """Create technical indicators and features"""
        print("Engineering features...")

        data = self.data.copy()

        # 1. Daily returns
        data['Daily_Return'] = data['Close'].pct_change()

        # 2. Moving averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()

        # 3. Volatility
        data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std()

        # 4. Price momentum
        data['Price_vs_MA20'] = data['Close'] / data['MA_20']
        data['Price_vs_MA50'] = data['Close'] / data['MA_50']

        # 5. Volume features
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']

        # 6. High-Low spread
        data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']

        # 7. Lagged prices
        data['Close_Lag1'] = data['Close'].shift(1)
        data['Close_Lag2'] = data['Close'].shift(2)
        data['Close_Lag3'] = data['Close'].shift(3)

        # 8. Target (next day's price)
        data['Target'] = data['Close'].shift(-1)

        # Define features
        self.features = ['Daily_Return', 'MA_5', 'MA_20', 'MA_50', 'Volatility_20', 
                        'Price_vs_MA20', 'Price_vs_MA50', 'Volume_MA_20', 'Volume_Ratio', 
                        'HL_Spread', 'Close_Lag1', 'Close_Lag2', 'Close_Lag3']

        # Clean data (remove NaN)
        self.clean_data = data.dropna()

        print(f"✓ Created {len(self.features)} features")
        print(f"✓ Clean dataset: {len(self.clean_data)} samples")

        return self.clean_data

    def prepare_data(self, test_size=0.2):
        """Split data for training and testing"""
        print("Preparing data for modeling...")

        # Remove target from features for prediction
        X = self.clean_data[self.features]
        y = self.clean_data['Target']

        # Time series split
        split_idx = int(len(self.clean_data) * (1 - test_size))

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, model_type='linear'):
        """Train the prediction model"""
        print(f"Training {model_type} model...")

        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

        # Train model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        # Calculate metrics
        self.train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        self.test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        self.test_mae = mean_absolute_error(self.y_test, test_pred)
        self.test_r2 = r2_score(self.y_test, test_pred)

        print(f"✓ Test RMSE: ${self.test_rmse:.2f}")
        print(f"✓ Test MAE: ${self.test_mae:.2f}")
        print(f"✓ Test R²: {self.test_r2:.4f}")

        return self.model

    def predict_next_day(self, date=None):
        """Predict the next trading day's price"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Get the most recent data for prediction
        latest_data = self.clean_data[self.features].iloc[-1:].copy()
        current_price = self.clean_data['Close'].iloc[-1]
        current_date = self.clean_data.index[-1]

        # Make prediction
        predicted_price = self.model.predict(latest_data)[0]
        price_change = predicted_price - current_price
        change_pct = (price_change / current_price) * 100

        print("\n" + "="*50)
        print("📈 NEXT DAY PRICE PREDICTION")
        print("="*50)
        print(f"Current date: {current_date.strftime('%Y-%m-%d')}")
        print(f"Current S&P 500 price: ${current_price:.2f}")
        print(f"Predicted next price: ${predicted_price:.2f}")
        print(f"Expected change: ${price_change:+.2f} ({change_pct:+.2f}%)")
        print("="*50)

        return {
            'current_date': current_date,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'change_percent': change_pct
        }

    def update_and_predict(self):
        """Download latest data and make a fresh prediction"""
        print("🔄 Updating with latest market data...")

        # Download most recent data
        self.download_data()
        self.engineer_features()
        self.prepare_data()
        self.train_model('linear')

        # Make prediction
        return self.predict_next_day()

# Example usage
if __name__ == "__main__":
    print("🚀 S&P 500 Price Predictor")
    print("=" * 40)

    # Initialize predictor
    predictor = SP500Predictor()

    # Download and prepare data
    predictor.download_data()
    predictor.engineer_features()
    predictor.prepare_data()

    # Train model
    predictor.train_model('linear')

    # Make prediction for next trading day
    prediction = predictor.predict_next_day()

    print("\n⚠️  DISCLAIMER: This is for educational purposes only.")
    print("   Never use this for actual trading decisions!")
