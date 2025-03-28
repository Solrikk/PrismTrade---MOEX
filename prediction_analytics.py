import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


class PredictionAnalytics:

    def __init__(self, prediction_dir="data/predictions"):
        self.prediction_dir = prediction_dir
        if not os.path.exists(self.prediction_dir):
            os.makedirs(self.prediction_dir)

    def get_prediction_files(self, ticker):
        ticker_dir = os.path.join(self.prediction_dir, ticker)
        if not os.path.exists(ticker_dir):
            return []

        return sorted(
            [f for f in os.listdir(ticker_dir) if f.endswith('.json')])

    def load_predictions(self, ticker):
        files = self.get_prediction_files(ticker)
        ticker_dir = os.path.join(self.prediction_dir, ticker)

        predictions = []
        for file in files:
            with open(os.path.join(ticker_dir, file), 'r') as f:
                data = json.load(f)
                predictions.append(data)

        return predictions

    def calculate_advanced_metrics(self, ticker):
        predictions = self.load_predictions(ticker)

        if len(predictions) < 5:
            return None

        intervals = ['15', '30', '60']
        results = {}

        for interval in intervals:
            prediction_actual_pairs = self.get_prediction_actual_pairs(
                predictions, interval)

            if len(prediction_actual_pairs) < 3:
                continue

            predicted_values = [
                pair['predicted'] for pair in prediction_actual_pairs
            ]
            actual_values = [
                pair['actual'] for pair in prediction_actual_pairs
            ]

            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mae = mean_absolute_error(actual_values, predicted_values)

            percentage_errors = [
                abs((pred - act) / act) * 100
                for pred, act in zip(predicted_values, actual_values)
            ]
            mape = np.mean(percentage_errors)

            directions_correct = 0
            for pair in prediction_actual_pairs:
                if (pair['predicted'] > pair['current_price'] and pair['actual'] > pair['current_price']) or \
                   (pair['predicted'] < pair['current_price'] and pair['actual'] < pair['current_price']):
                    directions_correct += 1

            direction_accuracy = (directions_correct /
                                  len(prediction_actual_pairs)) * 100

            error_chart_path = self.plot_error_distribution(
                percentage_errors, ticker, interval)

            results[interval] = {
                'rmse': round(rmse, 3),
                'mae': round(mae, 3),
                'mape': round(mape, 2),
                'direction_accuracy': round(direction_accuracy, 2),
                'samples': len(prediction_actual_pairs),
                'error_distribution_chart': error_chart_path
            }

        return results

    def get_prediction_actual_pairs(self, predictions, interval):
        pairs = []

        for i in range(len(predictions) - 1):
            current_pred = predictions[i]
            later_data = predictions[i + 1:]

            if interval not in current_pred['predictions']:
                continue

            minutes_diff = int(interval)
            target_time = datetime.fromisoformat(
                current_pred['timestamp']) + timedelta(minutes=minutes_diff)

            closest_idx = 0
            min_diff = timedelta(days=1)

            for j, future_data in enumerate(later_data):
                future_time = datetime.fromisoformat(future_data['timestamp'])
                time_diff = abs(future_time - target_time)

                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_idx = j

            if min_diff <= timedelta(minutes=5):
                actual_price = later_data[closest_idx]['current_price']
                predicted_price = current_pred['predictions'][interval][
                    'price']
                current_price = current_pred['current_price']

                pairs.append({
                    'timestamp':
                    current_pred['timestamp'],
                    'current_price':
                    current_price,
                    'predicted':
                    predicted_price,
                    'actual':
                    actual_price,
                    'error_pct':
                    abs((predicted_price - actual_price) / actual_price) * 100
                })

        return pairs

    def plot_error_distribution(self, percentage_errors, ticker, interval):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')

        plt.figure(figsize=(10, 6))
        plt.hist(percentage_errors, bins=10, alpha=0.7, color='#3498db')
        plt.axvline(np.mean(percentage_errors),
                    color='r',
                    linestyle='dashed',
                    linewidth=1)

        plt.title(
            f'Error Distribution for {ticker} ({interval}min predictions)')
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.annotate(f'Mean Error: {np.mean(percentage_errors):.2f}%',
                     xy=(0.7, 0.85),
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               alpha=0.8))

        plt.annotate(f'Median Error: {np.median(percentage_errors):.2f}%',
                     xy=(0.7, 0.78),
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               alpha=0.8))

        plt.annotate(f'Max Error: {np.max(percentage_errors):.2f}%',
                     xy=(0.7, 0.71),
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               alpha=0.8))

        chart_path = f'static/analytics/{ticker}_{interval}min_error_dist.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()

        return f'/{chart_path}'

    def perform_cross_validation(self, ticker, historical_prices, features):
        """
        Perform time series cross-validation on prediction models
        """
        if len(historical_prices) < 30:
            return None

        X = features
        y = historical_prices

        tscv = TimeSeriesSplit(n_splits=5)

        models = {
            'linear':
            LinearRegression(),
            'polynomial':
            Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('linear', LinearRegression())]),
            'gradient_boosting':
            GradientBoostingRegressor(n_estimators=50,
                                      learning_rate=0.1,
                                      max_depth=3,
                                      random_state=42)
        }

        cv_results = {}

        for model_name, model in models.items():
            rmse_scores = []
            mae_scores = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                rmse_scores.append(rmse)
                mae_scores.append(mae)

            cv_results[model_name] = {
                'avg_rmse': np.mean(rmse_scores),
                'avg_mae': np.mean(mae_scores),
                'rmse_scores': rmse_scores,
                'mae_scores': mae_scores
            }

        chart_path = self.plot_cv_results(cv_results, ticker)

        cv_summary = {
            'models': {},
            'chart_path':
            chart_path,
            'best_model':
            min(cv_results.items(), key=lambda x: x[1]['avg_rmse'])[0]
        }

        for model_name, results in cv_results.items():
            cv_summary['models'][model_name] = {
                'avg_rmse': round(results['avg_rmse'], 3),
                'avg_mae': round(results['avg_mae'], 3)
            }

        return cv_summary

    def plot_cv_results(self, cv_results, ticker):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')

        plt.figure(figsize=(12, 6))

        models = list(cv_results.keys())
        avg_rmse = [cv_results[model]['avg_rmse'] for model in models]

        bar_width = 0.35
        index = np.arange(len(models))

        plt.bar(index,
                avg_rmse,
                bar_width,
                alpha=0.8,
                color='#3498db',
                label='RMSE')

        plt.xlabel('Model')
        plt.ylabel('Error')
        plt.title(f'Cross-Validation Results for {ticker}')
        plt.xticks(index, models)
        plt.legend()

        # Add actual values
        for i, v in enumerate(avg_rmse):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')

        chart_path = f'static/analytics/{ticker}_cv_results.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()

        return f'/{chart_path}'

    def get_optimal_hyperparameters(self, ticker, historical_prices, features):
        """
        Perform hyperparameter optimization
        """
        if len(historical_prices) < 40:
            return None

        X = features
        y = historical_prices

        best_rmse = float('inf')
        best_params = {}

        n_estimators_options = [30, 50, 100]
        learning_rate_options = [0.05, 0.1, 0.2]
        max_depth_options = [2, 3, 4]

        results = []

        tscv = TimeSeriesSplit(n_splits=3)

        for n_estimators in n_estimators_options:
            for learning_rate in learning_rate_options:
                for max_depth in max_depth_options:
                    params = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth
                    }

                    rmse_scores = []

                    for train_idx, test_idx in tscv.split(X):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42)

                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)

                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        rmse_scores.append(rmse)

                    avg_rmse = np.mean(rmse_scores)
                    results.append({'params': params, 'avg_rmse': avg_rmse})

                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_params = params

        chart_path = self.plot_hyperparameter_results(results, ticker)

        return {
            'best_params': best_params,
            'best_rmse': round(best_rmse, 3),
            'chart_path': chart_path
        }

    def plot_hyperparameter_results(self, results, ticker):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')

        sorted_results = sorted(
            results, key=lambda x: x['avg_rmse'])[:10]  

        plt.figure(figsize=(12, 8))

        labels = [
            f"n={r['params']['n_estimators']}, lr={r['params']['learning_rate']}, d={r['params']['max_depth']}"
            for r in sorted_results
        ]
        rmse_values = [r['avg_rmse'] for r in sorted_results]

        plt.barh(range(len(labels)), rmse_values, color='#3498db', alpha=0.8)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('RMSE')
        plt.title(f'Top 10 Hyperparameter Combinations for {ticker}')
        plt.gca().invert_yaxis()

        for i, v in enumerate(rmse_values):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')

        chart_path = f'static/analytics/{ticker}_hyperparameters.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()

        return f'/{chart_path}'
