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

LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32
ARIMA_PARAMS_STATIONARY = (2, 0, 2)
ARIMA_PARAMS_NONSTATIONARY = (1, 1, 1)
ARIMA_FALLBACK_PARAMS = (1, 1, 0)

class PredictionAnalytics:
    def __init__(self, prediction_dir="data/predictions"):
        self.prediction_dir = prediction_dir
        if not os.path.exists(self.prediction_dir):
            os.makedirs(self.prediction_dir)

    def get_prediction_files(self, ticker):
        ticker_dir = os.path.join(self.prediction_dir, ticker)
        if not os.path.exists(ticker_dir):
            return []
        return sorted([f for f in os.listdir(ticker_dir) if f.endswith('.json')])

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
            prediction_actual_pairs = self.get_prediction_actual_pairs(predictions, interval)
            if len(prediction_actual_pairs) < 3:
                continue
            predicted_values = [pair['predicted'] for pair in prediction_actual_pairs]
            actual_values = [pair['actual'] for pair in prediction_actual_pairs]
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mae = mean_absolute_error(actual_values, predicted_values)
            percentage_errors = [abs((pred - act) / act) * 100 for pred, act in zip(predicted_values, actual_values)]
            mape = np.mean(percentage_errors)
            directions_correct = 0
            for pair in prediction_actual_pairs:
                if (pair['predicted'] > pair['current_price'] and pair['actual'] > pair['current_price']) or (pair['predicted'] < pair['current_price'] and pair['actual'] < pair['current_price']):
                    directions_correct += 1
            direction_accuracy = (directions_correct / len(prediction_actual_pairs)) * 100
            error_chart_path = self.plot_error_distribution(percentage_errors, ticker, interval)
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
            target_time = datetime.fromisoformat(current_pred['timestamp']) + timedelta(minutes=minutes_diff)
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
                predicted_price = current_pred['predictions'][interval]['price']
                current_price = current_pred['current_price']
                pairs.append({
                    'timestamp': current_pred['timestamp'],
                    'current_price': current_price,
                    'predicted': predicted_price,
                    'actual': actual_price,
                    'error_pct': abs((predicted_price - actual_price) / actual_price) * 100
                })
        return pairs

    def plot_error_distribution(self, percentage_errors, ticker, interval):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        plt.figure(figsize=(10, 6))
        plt.hist(percentage_errors, bins=10, alpha=0.7, color='#3498db')
        plt.axvline(np.mean(percentage_errors), color='r', linestyle='dashed', linewidth=1)
        plt.title(f'Error Distribution for {ticker} ({interval}min predictions)')
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.annotate(f'Mean Error: {np.mean(percentage_errors):.2f}%', xy=(0.7, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        plt.annotate(f'Median Error: {np.median(percentage_errors):.2f}%', xy=(0.7, 0.78), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        plt.annotate(f'Max Error: {np.max(percentage_errors):.2f}%', xy=(0.7, 0.71), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        chart_path = f'static/analytics/{ticker}_{interval}min_error_dist.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        return f'/{chart_path}'

    def perform_cross_validation(self, ticker, historical_prices, features):
        if len(historical_prices) < 30:
            return None
        X = features
        y = historical_prices
        tscv = TimeSeriesSplit(n_splits=5)
        models = {
            'linear': LinearRegression(),
            'polynomial': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
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
            'chart_path': chart_path,
            'best_model': min(cv_results.items(), key=lambda x: x[1]['avg_rmse'])[0]
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
        plt.bar(index, avg_rmse, bar_width, alpha=0.8, color='#3498db', label='RMSE')
        plt.xlabel('Model')
        plt.ylabel('Error')
        plt.title(f'Cross-Validation Results for {ticker}')
        plt.xticks(index, models)
        plt.legend()
        for i, v in enumerate(avg_rmse):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        chart_path = f'static/analytics/{ticker}_cv_results.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        return f'/{chart_path}'

    def get_optimal_hyperparameters(self, ticker, historical_prices, features):
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
                    params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}
                    rmse_scores = []
                    for train_idx, test_idx in tscv.split(X):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
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
        return {'best_params': best_params, 'best_rmse': round(best_rmse, 3), 'chart_path': chart_path}

    def plot_hyperparameter_results(self, results, ticker):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        sorted_results = sorted(results, key=lambda x: x['avg_rmse'])[:10]
        plt.figure(figsize=(12, 8))
        labels = [f"n={r['params']['n_estimators']}, lr={r['params']['learning_rate']}, d={r['params']['max_depth']}" for r in sorted_results]
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

    def evaluate_prediction_quality(self, ticker):
        predictions = self.load_predictions(ticker)
        if len(predictions) < 5:
            return {"error": "Недостаточно данных для анализа точности прогнозов"}
        learning_results = {
            'market_factors': {},
            'error_patterns': {},
            'improvement_factors': [],
            'model_adjustments': {},
            'meta_learning': {}
        }
        intervals = ['15', '30', '60']
        all_pairs = {}
        for interval in intervals:
            all_pairs[interval] = self.get_prediction_actual_pairs(predictions, interval)
            if len(all_pairs[interval]) < 3:
                continue
            error_magnitudes = []
            for pair in all_pairs[interval]:
                error_pct = pair.get('error_pct', 0)
                timestamp = pair.get('timestamp', '')
                current_price = pair.get('current_price', 0)
                predicted = pair.get('predicted', 0)
                actual = pair.get('actual', 0)
                prediction_diff = ((predicted - current_price) / current_price) * 100
                actual_diff = ((actual - current_price) / current_price) * 100
                error_magnitudes.append({
                    'timestamp': timestamp,
                    'error_pct': error_pct,
                    'prediction_diff': prediction_diff,
                    'actual_diff': actual_diff,
                    'error_ratio': abs(prediction_diff/actual_diff) if actual_diff != 0 else float('inf')
                })
            error_magnitudes.sort(key=lambda x: x['error_pct'], reverse=True)
            largest_errors = error_magnitudes[:min(5, len(error_magnitudes))]
            overestimation_count = sum(1 for em in error_magnitudes if em['prediction_diff'] > em['actual_diff'])
            underestimation_count = sum(1 for em in error_magnitudes if em['prediction_diff'] < em['actual_diff'])
            if overestimation_count > underestimation_count:
                bias_type = "переоценка"
                bias_ratio = overestimation_count / max(1, len(error_magnitudes))
            else:
                bias_type = "недооценка"
                bias_ratio = underestimation_count / max(1, len(error_magnitudes))
            learning_results['error_patterns'][interval] = {
                'bias_type': bias_type,
                'bias_ratio': round(bias_ratio * 100, 2),
                'largest_errors': [{
                    'timestamp': err['timestamp'],
                    'error_pct': round(err['error_pct'], 2),
                    'predicted_change': round(err['prediction_diff'], 2),
                    'actual_change': round(err['actual_diff'], 2)
                } for err in largest_errors]
            }
            prediction_sizes = [abs(em['prediction_diff']) for em in error_magnitudes]
            error_sizes = [em['error_pct'] for em in error_magnitudes]
            if len(prediction_sizes) > 3:
                correlation = np.corrcoef(prediction_sizes, error_sizes)[0, 1]
                learning_results['market_factors'][interval] = {
                    'prediction_error_correlation': round(correlation, 3),
                    'interpretation': ("Чем больше прогнозируемое изменение, тем больше ошибка" if correlation > 0.5 else "Величина прогноза слабо влияет на точность" if abs(correlation) < 0.3 else "Меньшие прогнозы имеют большую ошибку")
                }
                if correlation > 0.5:
                    learning_results['model_adjustments'][interval] = {
                        'reduce_magnitude': True,
                        'adjustment_factor': round(0.85 - 0.1 * min(correlation, 0.8), 2),
                        'explanation': "Рекомендуется уменьшить амплитуду прогнозов для повышения точности"
                    }
                elif correlation < -0.3:
                    learning_results['model_adjustments'][interval] = {
                        'increase_magnitude': True,
                        'adjustment_factor': round(1.15 + 0.1 * min(abs(correlation), 0.8), 2),
                        'explanation': "Рекомендуется увеличить амплитуду прогнозов для повышения точности"
                    }
            highly_volatile_errors = [em for em in error_magnitudes if abs(em['actual_diff']) > 0.8]
            if highly_volatile_errors:
                volatile_error_ratio = len(highly_volatile_errors) / len(error_magnitudes)
                learning_results['improvement_factors'].append({
                    'factor': 'volatility',
                    'impact': round(volatile_error_ratio * 100, 2),
                    'recommendation': ("Система должна корректировать прогнозы в моменты повышенной волатильности" if volatile_error_ratio > 0.3 else "Влияние волатильности на точность прогнозов несущественно")
                })
        if all(interval in learning_results['error_patterns'] for interval in intervals):
            interval_biases = [learning_results['error_patterns'][interval]['bias_type'] == "переоценка" for interval in intervals]
            consistent_bias = all(interval_biases) or not any(interval_biases)
            if consistent_bias:
                bias_direction = "переоценка" if interval_biases[0] else "недооценка"
                adjustment_value = 0.9 if bias_direction == "переоценка" else 1.1
                learning_results['meta_learning']['consistent_bias'] = {
                    'type': bias_direction,
                    'recommendation': f"Система систематически {bias_direction}ет изменение цены",
                    'global_adjustment_factor': adjustment_value
                }
            interval_accuracies = {}
            for interval in intervals:
                if interval in all_pairs and all_pairs[interval]:
                    direction_correct = 0
                    for pair in all_pairs[interval]:
                        pred_direction = pair['predicted'] > pair['current_price']
                        actual_direction = pair['actual'] > pair['current_price']
                        if pred_direction == actual_direction:
                            direction_correct += 1
                    interval_accuracies[interval] = direction_correct / len(all_pairs[interval])
            if interval_accuracies:
                best_interval = max(interval_accuracies.items(), key=lambda x: x[1])
                learning_results['meta_learning']['interval_performance'] = {
                    'best_interval': best_interval[0],
                    'accuracy': round(best_interval[1] * 100, 2),
                    'recommendation': f"Интервал {best_interval[0]} минут показывает наилучшую точность направления"
                }
        chart_path = self.plot_learning_curve(ticker, all_pairs)
        if chart_path:
            learning_results['learning_curve_chart'] = chart_path
        return learning_results

    def plot_learning_curve(self, ticker, all_pairs):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        has_data = False
        for interval in all_pairs:
            if len(all_pairs[interval]) >= 5:
                has_data = True
                break
        if not has_data:
            return None
        plt.figure(figsize=(12, 8))
        intervals = ['15', '30', '60']
        markers = ['o', 's', '^']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, interval in enumerate(intervals):
            if interval not in all_pairs or len(all_pairs[interval]) < 5:
                continue
            pairs = sorted(all_pairs[interval], key=lambda x: datetime.fromisoformat(x['timestamp']))
            window_size = min(3, len(pairs))
            errors = [p.get('error_pct', 0) for p in pairs]
            timestamps = [datetime.fromisoformat(p['timestamp']) for p in pairs]
            smoothed_errors = []
            for j in range(len(errors)):
                if j < window_size - 1:
                    smoothed_errors.append(np.mean(errors[:j+1]))
                else:
                    smoothed_errors.append(np.mean(errors[j-(window_size-1):j+1]))
            plt.plot(timestamps, smoothed_errors, marker=markers[i], color=colors[i], label=f'Ошибка прогноза {interval}мин', alpha=0.7, linestyle='-')
        plt.title(f'Динамика ошибок прогнозов для {ticker} (обучение системы)', fontsize=14)
        plt.ylabel('Процент ошибки (%)', fontsize=12)
        plt.xlabel('Время', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        max_interval = max(all_pairs.items(), key=lambda x: len(x[1]))[0]
        if max_interval and len(all_pairs[max_interval]) >= 5:
            pairs = sorted(all_pairs[max_interval], key=lambda x: datetime.fromisoformat(x['timestamp']))
            errors = [p.get('error_pct', 0) for p in pairs]
            x_numeric = np.arange(len(timestamps))
            if len(x_numeric) >= 5:
                try:
                    z = np.polyfit(x_numeric, errors, 1)
                    p_poly = np.poly1d(z)
                    plt.plot(timestamps, p_poly(x_numeric), "r--", linewidth=1.5, alpha=0.8, label=f'Тренд ошибки (интервал {max_interval}мин)')
                    if z[0] < -0.2:
                        plt.figtext(0.5, 0.01, "✅ Система улучшает точность прогнозов со временем", ha="center", fontsize=12, bbox={"facecolor":"green", "alpha":0.2, "pad":5})
                    elif z[0] > 0.2:
                        plt.figtext(0.5, 0.01, "⚠️ Система ухудшает точность прогнозов со временем", ha="center", fontsize=12, bbox={"facecolor":"red", "alpha":0.2, "pad":5})
                    else:
                        plt.figtext(0.5, 0.01, "ℹ️ Точность прогнозов стабильна со временем", ha="center", fontsize=12, bbox={"facecolor":"blue", "alpha":0.2, "pad":5})
                except Exception as e:
                    plt.plot(range(len(df)), df['error_pct'], marker='o', linestyle='-', color='#3498DB')
                    plt.title('Ошибка прогнозов')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        chart_path = f'static/analytics/{ticker}_learning_curve.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        return f'/{chart_path}'

    def build_lstm_model(self, ticker, historical_prices, features=None, intervals=['15', '30', '60']):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError:
            return {"error": "Требуется установить tensorflow для использования LSTM моделей", "recommendation": "Установите tensorflow с помощью команды: pip install tensorflow"}
        if len(historical_prices) < 50:
            return {"error": "Недостаточно исторических данных для обучения LSTM модели", "recommendation": "Необходимо минимум 50 точек данных"}
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(np.array(historical_prices).reshape(-1, 1))
        X_train, y_train = [], []
        time_steps = min(60, len(scaled_data) - 1)
        for i in range(time_steps, len(scaled_data)):
            X_train.append(scaled_data[i - time_steps:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        try:
            model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0)
        except Exception as e:
            return {"error": f"Ошибка при обучении LSTM модели: {str(e)}", "recommendation": "Попробуйте уменьшить размер батча или количество эпох"}
        predictions = {}
        for interval in intervals:
            steps_ahead = int(interval)
            last_sequence = scaled_data[-time_steps:]
            future_predictions = []
            current_batch = last_sequence.reshape((1, time_steps, 1))
            for i in range(steps_ahead):
                current_pred = model.predict(current_batch, verbose=0)[0]
                future_predictions.append(current_pred)
                new_sequence = np.append(current_batch[0, 1:, 0], current_pred)
                current_batch = new_sequence.reshape(1, time_steps, 1)
            unscaled_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            predicted_price = unscaled_predictions[-1][0]
            current_price = historical_prices[-1]
            change_percent = ((predicted_price - current_price) / current_price) * 100
            predictions[interval] = {
                'price': float(predicted_price),
                'change': float(change_percent),
                'all_points': [float(p[0]) for p in unscaled_predictions],
                'model_type': 'LSTM'
            }
        self.plot_lstm_comparison(ticker, historical_prices, predictions)
        return {
            'predictions': predictions,
            'model_info': {
                'type': 'LSTM',
                'layers': [
                    {"type": "LSTM", "units": 50, "return_sequences": True},
                    {"type": "Dropout", "rate": 0.2},
                    {"type": "LSTM", "units": 50, "return_sequences": False},
                    {"type": "Dropout", "rate": 0.2},
                    {"type": "Dense", "units": 1}
                ],
                'time_steps': time_steps,
                'training_samples': len(X_train)
            },
            'chart_path': f'/static/analytics/{ticker}_lstm_prediction.png'
        }

    def build_arima_model(self, ticker, historical_prices, intervals=['15', '30', '60']):
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            return {"error": "Требуется установить statsmodels для использования ARIMA моделей", "recommendation": "Установите statsmodels с помощью команды: pip install statsmodels"}
        if len(historical_prices) < 30:
            return {"error": "Недостаточно исторических данных для обучения ARIMA модели", "recommendation": "Необходимо минимум 30 точек данных"}
        result = adfuller(historical_prices)
        is_stationary = result[1] <= 0.05
        if is_stationary:
            p, d, q = ARIMA_PARAMS_STATIONARY
        else:
            p, d, q = ARIMA_PARAMS_NONSTATIONARY
        try:
            model = ARIMA(historical_prices, order=(p, d, q))
            model_fit = model.fit()
        except Exception as e:
            try:
                model = ARIMA(historical_prices, order=ARIMA_FALLBACK_PARAMS)
                model_fit = model.fit()
            except Exception as e2:
                return {"error": f"Ошибка при обучении ARIMA модели: {str(e2)}", "recommendation": "Попробуйте использовать другие параметры модели"}
        predictions = {}
        for interval in intervals:
            steps_ahead = int(interval)
            try:
                forecast = model_fit.forecast(steps=steps_ahead)
                predicted_price = forecast[-1]
                current_price = historical_prices[-1]
                change_percent = ((predicted_price - current_price) / current_price) * 100
                predictions[interval] = {
                    'price': float(predicted_price),
                    'change': float(change_percent),
                    'all_points': [float(p) for p in forecast],
                    'model_type': 'ARIMA',
                    'parameters': {'p': p, 'd': d, 'q': q}
                }
            except Exception as e:
                continue
        self.plot_arima_results(ticker, historical_prices, predictions, model_fit)
        return {
            'predictions': predictions,
            'model_info': {
                'type': 'ARIMA',
                'parameters': {'p': p, 'd': d, 'q': q},
                'is_stationary': is_stationary,
                'adf_pvalue': result[1]
            },
            'chart_path': f'/static/analytics/{ticker}_arima_prediction.png'
        }

    def plot_lstm_comparison(self, ticker, historical_prices, predictions):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        plt.figure(figsize=(12, 6))
        hist_len = min(len(historical_prices), 100)
        plt.plot(range(hist_len), historical_prices[-hist_len:], label='Исторические данные', color='#2C3E50')
        colors = {'15': '#3498DB', '30': '#2ECC71', '60': '#E74C3C'}
        current_price = historical_prices[-1]
        current_idx = hist_len - 1
        for interval, data in predictions.items():
            if 'all_points' in data:
                points = data['all_points']
                forecast_indices = range(current_idx, current_idx + len(points))
                plt.plot(forecast_indices, points, label=f'LSTM прогноз ({interval}м)', color=colors.get(interval, '#9B59B6'), linestyle='--')
                plt.scatter([forecast_indices[-1]], [points[-1]], color=colors.get(interval, '#9B59B6'), s=50)
                change_pct = data['change']
                change_text = f"+{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
                plt.annotate(change_text, xy=(forecast_indices[-1], points[-1]), xytext=(5, 5), textcoords="offset points", fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        plt.scatter([current_idx], [current_price], color='black', s=80, label='Текущая цена')
        plt.annotate(f"{current_price:.2f}", xy=(current_idx, current_price), xytext=(5, 5), textcoords="offset points", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        plt.title(f'LSTM прогноз для {ticker}', fontsize=14)
        plt.xlabel('Временные шаги', fontsize=12)
        plt.ylabel('Цена', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.figtext(0.5, 0.01, "Прогнозирование с помощью нейронной сети LSTM", ha="center", fontsize=10, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":5})
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'static/analytics/{ticker}_lstm_prediction.png', dpi=200)
        plt.close()

    def plot_arima_results(self, ticker, historical_prices, predictions, model_fit):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        hist_len = min(len(historical_prices), 100)
        plt.plot(range(hist_len), historical_prices[-hist_len:], label='Исторические данные', color='#2C3E50')
        current_price = historical_prices[-1]
        current_idx = hist_len - 1
        colors = {'15': '#3498DB', '30': '#2ECC71', '60': '#E74C3C'}
        for interval, data in predictions.items():
            if 'all_points' in data:
                points = data['all_points']
                forecast_indices = range(current_idx, current_idx + len(points))
                plt.plot(forecast_indices, points, label=f'ARIMA прогноз ({interval}м)', color=colors.get(interval, '#9B59B6'), linestyle='--')
                plt.scatter([forecast_indices[-1]], [points[-1]], color=colors.get(interval, '#9B59B6'), s=50)
        plt.scatter([current_idx], [current_price], color='black', s=80, label='Текущая цена')
        plt.title(f'ARIMA прогноз для {ticker}', fontsize=14)
        plt.xlabel('Временные шаги', fontsize=12)
        plt.ylabel('Цена', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.subplot(2, 1, 2)
        try:
            residuals = pd.DataFrame(model_fit.resid)
            residuals.plot(title='Остатки', ax=plt.gca(), color='#3498DB', legend=False)
            plt.xlabel('Временные шаги')
            plt.ylabel('Остатки')
            plt.grid(True, alpha=0.3)
            ax_inset = plt.axes([0.65, 0.25, 0.25, 0.2])
            residuals.hist(ax=ax_inset, bins=20, alpha=0.7, color='#3498DB')
            ax_inset.set_title('Распределение остатков')
            ax_inset.grid(False)
        except:
            plt.text(0.5, 0.5, 'Нет данных по остаткам модели', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        param_info = f"ARIMA({model_fit.model.order[0]},{model_fit.model.order[1]},{model_fit.model.order[2]})"
        plt.figtext(0.5, 0.01, f"Параметры модели: {param_info}", ha="center", fontsize=10, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":5})
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        chart_path = f'static/analytics/{ticker}_arima_prediction.png'
        plt.savefig(chart_path, dpi=200)
        plt.close()

    def combine_advanced_models(self, ticker, historical_prices, features=None):
        results = {}
        lstm_results = self.build_lstm_model(ticker, historical_prices, features)
        arima_results = self.build_arima_model(ticker, historical_prices)
        lstm_error = lstm_results.get('error', None)
        arima_error = arima_results.get('error', None)
        models_available = []
        if not lstm_error:
            models_available.append('lstm')
        if not arima_error:
            models_available.append('arima')
        if not models_available:
            return {"error": "Не удалось создать ни одну из продвинутых моделей", "lstm_error": lstm_error, "arima_error": arima_error}
        intervals = ['15', '30', '60']
        combined_predictions = {}
        for interval in intervals:
            preds = []
            weights = []
            if 'lstm' in models_available and interval in lstm_results.get('predictions', {}):
                lstm_pred = lstm_results['predictions'][interval]['price']
                preds.append(lstm_pred)
                weights.append(0.6)
            if 'arima' in models_available and interval in arima_results.get('predictions', {}):
                arima_pred = arima_results['predictions'][interval]['price']
                preds.append(arima_pred)
                weights.append(0.4)
            if preds:
                weights = [w / sum(weights) for w in weights]
                combined_price = sum(p * w for p, w in zip(preds, weights))
                current_price = historical_prices[-1]
                change_percent = ((combined_price - current_price) / current_price) * 100
                combined_predictions[interval] = {
                    'price': round(combined_price, 2),
                    'change': round(change_percent, 2),
                    'model_type': 'Ensemble (LSTM + ARIMA)',
                    'models_used': models_available,
                    'weights': dict(zip(models_available, weights))
                }
        chart_path = self.plot_model_comparison(ticker, historical_prices, lstm_results.get('predictions', {}), arima_results.get('predictions', {}), combined_predictions)
        return {
            'predictions': combined_predictions,
            'models_used': models_available,
            'model_details': {
                'lstm': lstm_results.get('model_info') if 'lstm' in models_available else None,
                'arima': arima_results.get('model_info') if 'arima' in models_available else None
            },
            'chart_path': chart_path
        }

    def plot_model_comparison(self, ticker, historical_prices, lstm_predictions, arima_predictions, combined_predictions):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        plt.figure(figsize=(12, 8))
        hist_len = min(len(historical_prices), 100)
        plt.plot(range(hist_len), historical_prices[-hist_len:], label='Исторические данные', color='#2C3E50')
        current_price = historical_prices[-1]
        current_idx = hist_len - 1
        interval = '15'
        if interval in lstm_predictions:
            lstm_price = lstm_predictions[interval]['price']
            plt.scatter([current_idx + int(interval)], [lstm_price], color='#3498DB', s=100, marker='o', label='LSTM')
            plt.plot([current_idx, current_idx + int(interval)], [current_price, lstm_price], color='#3498DB', linestyle='--')
        if interval in arima_predictions:
            arima_price = arima_predictions[interval]['price']
            plt.scatter([current_idx + int(interval)], [arima_price], color='#E74C3C', s=100, marker='s', label='ARIMA')
            plt.plot([current_idx, current_idx + int(interval)], [current_price, arima_price], color='#E74C3C', linestyle='--')
        if interval in combined_predictions:
            combined_price = combined_predictions[interval]['price']
            plt.scatter([current_idx + int(interval)], [combined_price], color='#2ECC71', s=140, marker='*', label='Ансамбль моделей')
            plt.plot([current_idx, current_idx + int(interval)], [current_price, combined_price], color='#2ECC71', linestyle='-', linewidth=2)
            change_pct = combined_predictions[interval]['change']
            change_text = f"+{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
            plt.annotate(f"Ансамбль: {combined_price:.2f} ({change_text})", xy=(current_idx + int(interval), combined_price), xytext=(10, 0), textcoords="offset points", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        plt.scatter([current_idx], [current_price], color='black', s=100, label='Текущая цена')
        plt.annotate(f"Текущая: {current_price:.2f}", xy=(current_idx, current_price), xytext=(10, -15), textcoords="offset points", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        for other_interval in ['30', '60']:
            plt.axvline(x=current_idx + int(other_interval), color='gray', linestyle=':', alpha=0.5)
            if other_interval in combined_predictions:
                price = combined_predictions[other_interval]['price']
                plt.scatter([current_idx + int(other_interval)], [price], color='#2ECC71', s=100, marker='*')
                plt.annotate(f"{other_interval}м: {price:.2f}", xy=(current_idx + int(other_interval), price), xytext=(5, 5), textcoords="offset points", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        plt.title(f'Сравнение моделей прогнозирования для {ticker}', fontsize=14)
        plt.xlabel('Временные шаги (минуты)', fontsize=12)
        plt.ylabel('Цена', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.figtext(0.5, 0.01, "Сравнение различных моделей машинного обучения для прогнозирования цен", ha="center", fontsize=10, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":5})
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        chart_path = f'static/analytics/{ticker}_model_comparison.png'
        plt.savefig(chart_path, dpi=200)
        plt.close()
        return f'/{chart_path}'

    def meta_learning(self, ticker):
        predictions = self.load_predictions(ticker)
        if len(predictions) < 10:
            return {"error": "Недостаточно данных для метаобучения", "recommendation": "Необходимо минимум 10 прогнозов в истории"}
        meta_data = {'15': [], '30': [], '60': []}
        for interval in meta_data.keys():
            prediction_actual_pairs = self.get_prediction_actual_pairs(predictions, interval)
            if len(prediction_actual_pairs) < 5:
                continue
            for pair in prediction_actual_pairs:
                original_prediction = None
                for pred in predictions:
                    if pred['timestamp'] == pair['timestamp'] and interval in pred.get('predictions', {}):
                        original_prediction = pred
                        break
                if not original_prediction:
                    continue
                prediction_diff = ((pair['predicted'] - pair['current_price']) / pair['current_price']) * 100
                actual_diff = ((pair['actual'] - pair['current_price']) / pair['current_price']) * 100
                error = prediction_diff - actual_diff
                market_state = original_prediction.get('market_state', {})
                meta_features = {
                    'timestamp': pair['timestamp'],
                    'current_price': pair['current_price'],
                    'predicted_price': pair['predicted'],
                    'actual_price': pair['actual'],
                    'prediction_diff_pct': prediction_diff,
                    'actual_diff_pct': actual_diff,
                    'error_pct': pair['error_pct'],
                    'error_direction': 1 if prediction_diff > actual_diff else -1,
                    'is_bullish': market_state.get('bullish', False),
                    'is_bearish': market_state.get('bearish', False),
                    'trend_strength': market_state.get('trend_strength', 50),
                    'was_overbought': market_state.get('overbought', False),
                    'was_oversold': market_state.get('oversold', False),
                    'prediction_magnitude': abs(prediction_diff),
                    'market_volatility': original_prediction.get('volatility', 1.0)
                }
                meta_data[interval].append(meta_features)
        meta_learning_results = {}
        for interval, data in meta_data.items():
            if len(data) < 5:
                continue
            df = pd.DataFrame(data)
            key_features = ['prediction_diff_pct', 'prediction_magnitude', 'market_volatility', 'trend_strength']
            for bool_feature in ['is_bullish', 'is_bearish', 'was_overbought', 'was_oversold']:
                df[bool_feature] = df[bool_feature].astype(int)
                key_features.append(bool_feature)
            correlations = {}
            for feature in key_features:
                if feature in df.columns:
                    correlations[feature] = df[feature].corr(df['error_pct'])
            bias = df['error_pct'].mean()
            correction_rules = []
            if abs(bias) > 0.5:
                correction_rules.append({
                    'type': 'global_bias',
                    'description': f"Систематическая ошибка {bias:.2f}%",
                    'adjustment': -bias,
                    'confidence': min(100, max(50, 50 + abs(bias) * 10))
                })
            for feature, corr in correlations.items():
                if abs(corr) > 0.3:
                    sign = "положительная" if corr > 0 else "отрицательная"
                    explanation = f"При увеличении {feature} ошибка {sign}, коэффициент {corr:.2f}"
                    correction_rules.append({
                        'type': 'feature_correlation',
                        'feature': feature,
                        'correlation': corr,
                        'description': explanation,
                        'adjustment_multiplier': -corr * 0.5,
                        'confidence': min(90, max(60, 60 + abs(corr) * 30))
                    })
            market_states = [
                ('is_bullish', 1, 'в бычьем тренде'),
                ('is_bearish', 1, 'в медвежьем тренде'),
                ('was_overbought', 1, 'при перекупленности'),
                ('was_oversold', 1, 'при перепроданности')
            ]
            for feature, value, description in market_states:
                if feature in df.columns:
                    specific_state_df = df[df[feature] == value]
                    if len(specific_state_df) >= 3:
                        state_bias = specific_state_df['error_pct'].mean()
                        if abs(state_bias) > 1.0:
                            correction_rules.append({
                                'type': 'market_state',
                                'feature': feature,
                                'description': f"Систематическая ошибка {state_bias:.2f}% {description}",
                                'adjustment': -state_bias,
                                'condition': {feature: value},
                                'confidence': min(95, max(60, 60 + abs(state_bias) * 5))
                            })
            X = df[key_features].fillna(0)
            y = df['error_pct']
            correction_model = None
            model_description = "Не удалось создать модель коррекции"
            model_score = 0
            if len(X) >= 10:
                try:
                    from sklearn.linear_model import LinearRegression, Ridge
                    from sklearn.ensemble import GradientBoostingRegressor
                    from sklearn.model_selection import train_test_split
                    models = {
                        'linear': LinearRegression(),
                        'ridge': Ridge(alpha=1.0),
                        'gbm': GradientBoostingRegressor(n_estimators=30, max_depth=3)
                    }
                    best_model = None
                    best_score = -float('inf')
                    for name, model in models.items():
                        try:
                            model.fit(X, y)
                            score = model.score(X, y)
                            if score > best_score:
                                best_score = score
                                best_model = (name, model)
                        except:
                            continue
                    if best_model:
                        correction_model = best_model[1]
                        model_description = f"Модель {best_model[0]}, R² = {best_score:.3f}"
                        model_score = best_score
                except Exception as e:
                    correction_model = None
                    model_description = f"Ошибка при создании модели: {str(e)}"
            chart_path = self.plot_meta_learning_analysis(ticker, interval, df)
            meta_learning_results[interval] = {
                'sample_size': len(data),
                'bias': bias,
                'correlations': correlations,
                'correction_rules': correction_rules,
                'model_description': model_description,
                'model_score': model_score,
                'has_correction_model': correction_model is not None,
                'chart_path': chart_path
            }
            if correction_model is not None:
                meta_learning_results[interval]['model'] = 'correction_model_object'
        return meta_learning_results

    def plot_meta_learning_analysis(self, ticker, interval, df):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        if 'timestamp' in df.columns and 'error_pct' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df_sorted = df.sort_values('datetime')
                plt.plot(range(len(df_sorted)), df_sorted['error_pct'], marker='o', linestyle='-', color='#3498DB')
                z = np.polyfit(range(len(df_sorted)), df_sorted['error_pct'], 1)
                p_poly = np.poly1d(z)
                plt.plot(range(len(df_sorted)), p_poly(range(len(df_sorted))), "r--", linewidth=1.5, alpha=0.8)
                trend_direction = "улучшается" if z[0] < 0 else "ухудшается"
                plt.title(f'Ошибка прогнозов со временем (тренд {trend_direction})')
            except:
                plt.plot(range(len(df)), df['error_pct'], marker='o', linestyle='-', color='#3498DB')
                plt.title('Ошибка прогнозов')
        else:
            plt.text(0.5, 0.5, 'Недостаточно данных', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Ошибка прогнозов')
        plt.ylabel('Ошибка, %')
        plt.xlabel('Номер прогноза')
        plt.axhline(y=0, color='green', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 2)
        if 'prediction_diff_pct' in df.columns and 'error_pct' in df.columns:
            plt.scatter(df['prediction_diff_pct'], df['error_pct'], alpha=0.7, color='#E74C3C')
            if len(df) >= 3:
                try:
                    z = np.polyfit(df['prediction_diff_pct'], df['error_pct'], 1)
                    p_poly = np.poly1d(z)
                    x_range = np.linspace(df['prediction_diff_pct'].min(), df['prediction_diff_pct'].max(), 100)
                    plt.plot(x_range, p_poly(x_range), "g--", linewidth=1.5, alpha=0.8)
                    corr = df['prediction_diff_pct'].corr(df['error_pct'])
                    plt.title(f'Размер прогноза vs Ошибка (r={corr:.2f})')
                except:
                    plt.title('Размер прогноза vs Ошибка')
            else:
                plt.title('Размер прогноза vs Ошибка')
        else:
            plt.text(0.5, 0.5, 'Недостаточно данных', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Размер прогноза vs Ошибка')
        plt.ylabel('Ошибка, %')
        plt.xlabel('Прогнозируемое изменение, %')
        plt.axhline(y=0, color='green', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='green', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        if 'error_pct' in df.columns and len(df) >= 3:
            plt.hist(df['error_pct'], bins=min(10, len(df)), alpha=0.7, color='#9B59B6')
            plt.axvline(df['error_pct'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Среднее: {df["error_pct"].mean():.2f}%')
            plt.axvline(df['error_pct'].median(), color='green', linestyle='dashed', linewidth=2, label=f'Медиана: {df["error_pct"].median():.2f}%')
            plt.title('Распределение ошибок')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Недостаточно данных', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Распределение ошибок')
        plt.ylabel('Частота')
        plt.xlabel('Ошибка, %')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 4)
        try:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) >= 3:
                corr_matrix = df[numeric_cols].corr()
                im = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
                plt.colorbar(im, orientation='vertical', shrink=0.8)
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        color = "white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
                        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
                plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90, fontsize=8)
                plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=8)
                plt.title('Корреляционная матрица')
            else:
                plt.text(0.5, 0.5, 'Недостаточно числовых признаков', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.title('Корреляционная матрица')
        except Exception as e:
            plt.text(0.5, 0.5, f'Ошибка построения: {str(e)}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8)
            plt.title('Корреляционная матрица')
        plt.tight_layout()
        plt.suptitle(f'Метаобучение для {ticker} (интервал {interval} мин)', fontsize=16, y=1.02)
        chart_path = f'static/analytics/{ticker}_meta_learning_{interval}.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        return f'/{chart_path}'

    def apply_meta_learning_corrections(self, ticker, predictions):
        meta_learning_results = self.meta_learning(ticker)
        if isinstance(meta_learning_results, dict) and 'error' in meta_learning_results:
            return predictions, {'applied': False, 'reason': meta_learning_results['error'], 'original_predictions': predictions}
        corrected_predictions = {}
        correction_details = {'intervals': {}}
        for interval, pred_data in predictions.items():
            if interval in meta_learning_results:
                meta_data = meta_learning_results[interval]
                original_price = pred_data['price']
                original_change = pred_data['change']
                total_adjustment_pct = 0
                adjustment_reasons = []
                for rule in meta_data.get('correction_rules', []):
                    if rule['type'] == 'global_bias':
                        total_adjustment_pct += rule['adjustment']
                        adjustment_reasons.append({'type': 'global_bias', 'description': rule['description'], 'adjustment': rule['adjustment'], 'confidence': rule['confidence']})
                    elif rule['type'] == 'feature_correlation' and 'feature' in rule:
                        feature = rule['feature']
                        if feature == 'prediction_diff_pct':
                            feature_value = original_change
                            adjustment = feature_value * rule['adjustment_multiplier']
                            total_adjustment_pct += adjustment
                            adjustment_reasons.append({'type': 'feature_correlation', 'feature': feature, 'feature_value': feature_value, 'adjustment': adjustment, 'confidence': rule['confidence']})
                        elif feature == 'market_volatility' and 'volatility' in predictions:
                            feature_value = predictions['volatility']
                            adjustment = feature_value * rule['adjustment_multiplier']
                            total_adjustment_pct += adjustment
                            adjustment_reasons.append({'type': 'feature_correlation', 'feature': feature, 'feature_value': feature_value, 'adjustment': adjustment, 'confidence': rule['confidence']})
                if abs(total_adjustment_pct) > 0.01:
                    corrected_change = original_change + total_adjustment_pct
                    current_price = predictions.get('current_price', original_price / (1 + original_change / 100))
                    corrected_price = current_price * (1 + corrected_change / 100)
                    corrected_pred = pred_data.copy()
                    corrected_pred['price'] = corrected_price
                    corrected_pred['change'] = corrected_change
                    corrected_pred['meta_learning_applied'] = True
                    corrected_pred['original_price'] = original_price
                    corrected_pred['original_change'] = original_change
                    corrected_pred['adjustment_pct'] = total_adjustment_pct
                    corrected_predictions[interval] = corrected_pred
                    correction_details['intervals'][interval] = {'applied': True, 'original_price': original_price, 'corrected_price': corrected_price, 'adjustment_pct': total_adjustment_pct, 'adjustment_reasons': adjustment_reasons, 'confidence': meta_data.get('model_score', 0) * 100}
                else:
                    corrected_predictions[interval] = pred_data
                    correction_details['intervals'][interval] = {'applied': False, 'reason': 'Незначительная коррекция', 'adjustment_pct': total_adjustment_pct}
            else:
                corrected_predictions[interval] = pred_data
                correction_details['intervals'][interval] = {'applied': False, 'reason': 'Нет данных метаобучения для этого интервала'}
        correction_details['applied'] = any(details.get('applied', False) for details in correction_details['intervals'].values())
        correction_details['original_predictions'] = predictions
        chart_path = self.plot_meta_learning_corrections(ticker, predictions, corrected_predictions, correction_details)
        correction_details['chart_path'] = chart_path
        return corrected_predictions, correction_details

    def plot_meta_learning_corrections(self, ticker, original_predictions, corrected_predictions, correction_details):
        if not os.path.exists('static/analytics'):
            os.makedirs('static/analytics')
        plt.figure(figsize=(12, 8))
        current_price = original_predictions.get('current_price', None)
        if not current_price and '15' in original_predictions:
            original_price = original_predictions['15'].get('price', 100)
            original_change = original_predictions['15'].get('change', 0)
            if original_change != 0:
                current_price = original_price / (1 + original_change / 100)
            else:
                current_price = 100
        x_labels = ['Текущая'] + [f'{i} мин' for i in sorted([int(i) for i in original_predictions.keys() if i.isdigit()])]
        x_positions = list(range(len(x_labels)))
        original_values = [current_price]
        corrected_values = [current_price]
        intervals = sorted([i for i in original_predictions.keys() if i.isdigit()], key=int)
        for interval in intervals:
            if interval in original_predictions:
                original_values.append(original_predictions[interval]['price'])
            else:
                original_values.append(None)
            if interval in corrected_predictions:
                corrected_values.append(corrected_predictions[interval]['price'])
            else:
                corrected_values.append(None)
        plt.plot(x_positions, original_values, 'o-', label='Исходный прогноз', color='#3498DB', linewidth=2)
        plt.plot(x_positions, corrected_values, 'o-', label='Скорректированный прогноз', color='#E74C3C', linewidth=2)
        for i, interval in enumerate(intervals, 1):
            if interval in original_predictions and interval in corrected_predictions:
                original_price = original_predictions[interval]['price']
                corrected_price = corrected_predictions[interval]['price']
                if abs(original_price - corrected_price) > 0.01:
                    plt.plot([x_positions[i], x_positions[i]], [original_price, corrected_price], 'k--', alpha=0.5)
                    adjustment_pct = ((corrected_price - original_price) / original_price) * 100
                    adjustment_text = f"+{adjustment_pct:.2f}%" if adjustment_pct >= 0 else f"{adjustment_pct:.2f}%"
                    if corrected_price > original_price:
                        ytext = (original_price + corrected_price) / 2
                        xytext = (5, 0)
                    else:
                        ytext = (original_price + corrected_price) / 2
                        xytext = (5, 0)
                    plt.annotate(adjustment_text, xy=(x_positions[i], ytext), xytext=xytext, textcoords="offset points", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", alpha=0.7))
        for i, (original, corrected) in enumerate(zip(original_values, corrected_values)):
            if original is not None:
                plt.annotate(f"{original:.2f}", xy=(x_positions[i], original), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, color='#3498DB')
            if corrected is not None and abs(original - corrected) > 0.01:
                plt.annotate(f"{corrected:.2f}", xy=(x_positions[i], corrected), xytext=(0, -15), textcoords="offset points", ha='center', fontsize=9, color='#E74C3C')
        plt.title(f'Метаобучение: коррекция прогнозов для {ticker}', fontsize=14)
        plt.xlabel('Интервал прогнозирования', fontsize=12)
        plt.ylabel('Цена', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.xticks(x_positions, x_labels)
        explanation_text = "Примененные правила коррекции:\n"
        has_corrections = False
        for interval, details in correction_details.get('intervals', {}).items():
            if details.get('applied', False):
                has_corrections = True
                explanation_text += f"\nИнтервал {interval} мин: {details.get('adjustment_pct', 0):.2f}%\n"
                for i, reason in enumerate(details.get('adjustment_reasons', [])[:2]):
                    explanation_text += f"  - {reason.get('description', '')}\n"
                if len(details.get('adjustment_reasons', [])) > 2:
                    explanation_text += f"  - ... и еще {len(details.get('adjustment_reasons', [])) - 2} правил\n"
        if not has_corrections:
            explanation_text += "\nНе было применено значимых корректировок."
        plt.figtext(0.5, 0.01, explanation_text, ha="center", va="bottom", fontsize=9, bbox={"facecolor":"#f0f0f0", "alpha":0.8, "pad":5})
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        chart_path = f'static/analytics/{ticker}_meta_learning_corrections.png'
        plt.savefig(chart_path, dpi=200)
        plt.close()
        return f'/{chart_path}'
