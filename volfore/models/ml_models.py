"""
Machine learning models for volatility forecasting.

This module implements various machine learning approaches for volatility forecasting:
- Random Forest
- XGBoost
- LSTM Neural Networks
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class MLVolatilityModel:
    """
    Base class for machine learning volatility models.
    
    Provides common functionality for feature engineering,
    training, and evaluation of ML-based volatility forecasting models.
    """
    
    def __init__(self, feature_window=22, target_window=1):
        """
        Initialize ML volatility model.
        
        Parameters
        ----------
        feature_window : int, default=22
            Number of past days to use for feature generation
        target_window : int, default=1
            Number of days ahead to forecast
        """
        self.feature_window = feature_window
        self.target_window = target_window
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        
    def _engineer_features(self, returns, realized_vol, additional_features=None):
        """
        Create features for volatility prediction.
        
        Parameters
        ----------
        returns : Series or array-like
            Asset returns
        realized_vol : Series or array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features like VIX, volume, etc.
            
        Returns
        -------
        tuple
            X (features) and y (target) for model training
        """
        # Convert inputs to pandas Series if they aren't already
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        if not isinstance(realized_vol, pd.Series):
            realized_vol = pd.Series(realized_vol, index=returns.index)
            
        # Create DataFrame for features
        df = pd.DataFrame(index=returns.index)
        
        # Add basic features
        df['return'] = returns
        df['abs_return'] = np.abs(returns)
        df['return_squared'] = returns**2
        df['realized_vol'] = realized_vol
        
        # Add lagged returns
        for lag in range(1, min(22, self.feature_window) + 1):
            df[f'return_lag_{lag}'] = returns.shift(lag)
            df[f'abs_return_lag_{lag}'] = np.abs(returns).shift(lag)
            
        # Add rolling window features
        windows = [5, 10, 22]
        for window in windows:
            if window <= self.feature_window:
                df[f'return_mean_{window}d'] = returns.rolling(window=window).mean().shift(1)
                df[f'return_std_{window}d'] = returns.rolling(window=window).std().shift(1)
                df[f'abs_return_mean_{window}d'] = np.abs(returns).rolling(window=window).mean().shift(1)
                df[f'return_squared_mean_{window}d'] = (returns**2).rolling(window=window).mean().shift(1)
        
        # Add lagged realized volatility
        for lag in range(1, min(22, self.feature_window) + 1):
            df[f'realized_vol_lag_{lag}'] = realized_vol.shift(lag)
        
        # Add rolling realized volatility
        for window in windows:
            if window <= self.feature_window:
                df[f'realized_vol_mean_{window}d'] = realized_vol.rolling(window=window).mean().shift(1)
        
        # Add volatility of volatility
        for window in windows:
            if window <= self.feature_window:
                df[f'realized_vol_std_{window}d'] = realized_vol.rolling(window=window).std().shift(1)
        
        # Add target: future realized volatility
        if self.target_window == 1:
            df['target'] = realized_vol.shift(-1)
        else:
            # Average volatility over target_window
            df['target'] = realized_vol.rolling(window=self.target_window).mean().shift(-self.target_window)
        
        # Add additional features if provided
        if additional_features is not None:
            # Ensure index alignment
            additional_features = additional_features.reindex(df.index)
            
            # Add each additional feature
            for col in additional_features.columns:
                df[col] = additional_features[col]
                
                # Add lagged versions for time-varying features
                for lag in range(1, min(5, self.feature_window) + 1):
                    df[f'{col}_lag_{lag}'] = additional_features[col].shift(lag)
        
        # Drop NaN rows
        df = df.dropna()
        
        # Store feature names (all columns except 'target')
        self.feature_names = [col for col in df.columns if col != 'target']
        
        # Split into X and y
        X = df[self.feature_names]
        y = df['target']
        
        return X, y
    
    def fit(self, returns, realized_vol, additional_features=None, **kwargs):
        """
        Train the model.
        
        This is an abstract method to be implemented by subclasses.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
        **kwargs
            Additional keyword arguments for model training
            
        Returns
        -------
        self : object
            Returns self
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, returns, realized_vol, additional_features=None):
        """
        Predict volatility using trained model.
        
        This is an abstract method to be implemented by subclasses.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
            
        Returns
        -------
        predictions : ndarray
            Predicted volatility
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def evaluate(self, returns, realized_vol, additional_features=None, metrics=None):
        """
        Evaluate model performance.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
        metrics : list, optional
            List of metrics to compute
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
            
        # Get predictions
        predictions = self.predict(returns, realized_vol, additional_features)
        
        # Get actual values
        X, y_true = self._engineer_features(returns, realized_vol, additional_features)
        
        # Ensure length match
        min_length = min(len(predictions), len(y_true))
        predictions = predictions[-min_length:]
        y_true = y_true.values[-min_length:]
        
        # Calculate metrics
        results = {}
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, predictions))
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_true, predictions)
        if 'r2' in metrics:
            results['r2'] = r2_score(y_true, predictions)
        if 'mape' in metrics:
            # Avoid division by zero
            mask = y_true != 0
            results['mape'] = np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100
            
        return results
    
    def feature_importance(self):
        """
        Get feature importance if available.
        
        This is an abstract method to be implemented by subclasses.
        
        Returns
        -------
        DataFrame or None
            Feature importance if available
        """
        return None


class RandomForestVolModel(MLVolatilityModel):
    """
    Random Forest model for volatility forecasting.
    """
    
    def __init__(self, feature_window=22, target_window=1, n_estimators=100, 
                 max_depth=None, min_samples_split=2, random_state=42):
        """
        Initialize Random Forest volatility model.
        
        Parameters
        ----------
        feature_window : int, default=22
            Number of past days to use for feature generation
        target_window : int, default=1
            Number of days ahead to forecast
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int, default=2
            Minimum samples required to split an internal node
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__(feature_window, target_window)
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
    
    def fit(self, returns, realized_vol, additional_features=None, scale=True, **kwargs):
        """
        Train Random Forest model.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
        scale : bool, default=True
            Whether to scale features
        **kwargs
            Additional kwargs passed to the model's fit method
            
        Returns
        -------
        self : object
            Returns self
        """
        # Prepare features
        X, y = self._engineer_features(returns, realized_vol, additional_features)
        
        # Scale if requested
        if scale:
            X = pd.DataFrame(
                self.scaler_X.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            y = pd.Series(
                self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index
            )
        
        # Train model
        self.model.fit(X, y, **kwargs)
        
        return self
    
    def predict(self, returns, realized_vol, additional_features=None):
        """
        Predict volatility using Random Forest model.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
            
        Returns
        -------
        predictions : ndarray
            Predicted volatility
        """
        # Prepare features
        X, _ = self._engineer_features(returns, realized_vol, additional_features)
        
        # Scale features
        X = pd.DataFrame(
            self.scaler_X.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Make predictions
        preds = self.model.predict(X)
        
        # Inverse transform if needed
        preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        
        return preds
    
    def feature_importance(self):
        """
        Get feature importance for Random Forest model.
        
        Returns
        -------
        DataFrame
            Feature importance
        """
        if self.model is None or self.feature_names is None:
            return None
            
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        fi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        fi_df = fi_df.sort_values('Importance', ascending=False)
        
        return fi_df


class XGBoostVolModel(MLVolatilityModel):
    """
    XGBoost model for volatility forecasting.
    """
    
    def __init__(self, feature_window=22, target_window=1, n_estimators=100, 
                 learning_rate=0.1, max_depth=6, subsample=0.8, 
                 colsample_bytree=0.8, random_state=42):
        """
        Initialize XGBoost volatility model.
        
        Parameters
        ----------
        feature_window : int, default=22
            Number of past days to use for feature generation
        target_window : int, default=1
            Number of days ahead to forecast
        n_estimators : int, default=100
            Number of boosting rounds
        learning_rate : float, default=0.1
            Learning rate
        max_depth : int, default=6
            Maximum tree depth
        subsample : float, default=0.8
            Subsample ratio of training instances
        colsample_bytree : float, default=0.8
            Subsample ratio of columns when constructing each tree
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__(feature_window, target_window)
        
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )
    
    def fit(self, returns, realized_vol, additional_features=None, 
            scale=True, early_stopping=True, validation_size=0.2, **kwargs):
        """
        Train XGBoost model.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
        scale : bool, default=True
            Whether to scale features
        early_stopping : bool, default=True
            Whether to use early stopping
        validation_size : float, default=0.2
            Fraction of data to use for validation if early_stopping=True
        **kwargs
            Additional kwargs passed to the model's fit method
            
        Returns
        -------
        self : object
            Returns self
        """
        # Prepare features
        X, y = self._engineer_features(returns, realized_vol, additional_features)
        
        # Scale if requested
        if scale:
            X = pd.DataFrame(
                self.scaler_X.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            y = pd.Series(
                self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index
            )
        
        # Setup for early stopping if requested
        eval_set = None
        if early_stopping:
            # Time-based split for validation
            split_idx = int(len(X) * (1 - validation_size))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            eval_set = [(X_val, y_val)]
            
            # Train model with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False,
                **kwargs
            )
        else:
            # Train model without early stopping
            self.model.fit(X, y, **kwargs)
        
        return self
    
    def predict(self, returns, realized_vol, additional_features=None):
        """
        Predict volatility using XGBoost model.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
            
        Returns
        -------
        predictions : ndarray
            Predicted volatility
        """
        # Prepare features
        X, _ = self._engineer_features(returns, realized_vol, additional_features)
        
        # Scale features
        X = pd.DataFrame(
            self.scaler_X.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Make predictions
        preds = self.model.predict(X)
        
        # Inverse transform if needed
        preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        
        return preds
    
    def feature_importance(self, importance_type='gain'):
        """
        Get feature importance for XGBoost model.
        
        Parameters
        ----------
        importance_type : str, default='gain'
            Importance type, one of 'weight', 'gain', 'cover', 'total_gain', 'total_cover'
            
        Returns
        -------
        DataFrame
            Feature importance
        """
        if self.model is None or self.feature_names is None:
            return None
            
        # Get feature importance
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Create DataFrame (handle missing features)
        features = []
        scores = []
        
        for feature in self.feature_names:
            features.append(feature)
            scores.append(importance.get(feature, 0))
        
        fi_df = pd.DataFrame({
            'Feature': features,
            'Importance': scores
        })
        
        # Sort by importance
        fi_df = fi_df.sort_values('Importance', ascending=False)
        
        return fi_df


class LSTMVolModel(MLVolatilityModel):
    """
    LSTM Neural Network model for volatility forecasting.
    """
    
    def __init__(self, feature_window=22, target_window=1, units=50, 
                 dropout=0.1, recurrent_dropout=0.1, learning_rate=0.001):
        """
        Initialize LSTM volatility model.
        
        Parameters
        ----------
        feature_window : int, default=22
            Number of past days to use for feature generation (lookback period)
        target_window : int, default=1
            Number of days ahead to forecast
        units : int, default=50
            Number of LSTM units
        dropout : float, default=0.1
            Dropout rate
        recurrent_dropout : float, default=0.1
            Recurrent dropout rate
        learning_rate : float, default=0.001
            Learning rate for optimizer
        """
        super().__init__(feature_window, target_window)
        
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.model = None
        self.lookback = feature_window  # Lookback period for sequence data
    
    def _create_sequences(self, X, y):
        """
        Create sequences for LSTM model.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            Features
        y : Series or ndarray
            Target
            
        Returns
        -------
        tuple
            Sequences X and y
        """
        X_seq = []
        y_seq = []
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Create sequences
        for i in range(len(X) - self.lookback + 1):
            X_seq.append(X[i:i+self.lookback])
            y_seq.append(y[i+self.lookback-1])
            
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, input_shape):
        """
        Build LSTM model.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input data (lookback, n_features)
            
        Returns
        -------
        Model
            Compiled Keras model
        """
        model = Sequential()
        
        # LSTM layer
        model.add(LSTM(
            units=self.units,
            input_shape=input_shape,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            return_sequences=False
        ))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        
        return model
    
    def fit(self, returns, realized_vol, additional_features=None, 
            scale=True, epochs=100, batch_size=32, validation_split=0.2, 
            early_stopping=True, **kwargs):
        """
        Train LSTM model.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
        scale : bool, default=True
            Whether to scale features
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size
        validation_split : float, default=0.2
            Fraction of data to use for validation
        early_stopping : bool, default=True
            Whether to use early stopping
        **kwargs
            Additional kwargs passed to the model's fit method
            
        Returns
        -------
        self : object
            Returns self
        """
        # Prepare features
        X, y = self._engineer_features(returns, realized_vol, additional_features)
        
        # Scale if requested
        if scale:
            X = pd.DataFrame(
                self.scaler_X.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            y = pd.Series(
                self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index
            )
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        # Setup callbacks
        callbacks = []
        if early_stopping:
            es = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(es)
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
            **kwargs
        )
        
        self.history = history
        
        return self
    
    def predict(self, returns, realized_vol, additional_features=None):
        """
        Predict volatility using LSTM model.
        
        Parameters
        ----------
        returns : array-like
            Asset returns
        realized_vol : array-like
            Realized volatility
        additional_features : DataFrame, optional
            Additional features
            
        Returns
        -------
        predictions : ndarray
            Predicted volatility
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Prepare features
        X, _ = self._engineer_features(returns, realized_vol, additional_features)
        
        # Scale features
        X = pd.DataFrame(
            self.scaler_X.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Create sequences
        X_seq = []
        for i in range(len(X) - self.lookback + 1):
            X_seq.append(X.values[i:i+self.lookback])
        X_seq = np.array(X_seq)
        
        # Make predictions
        preds = self.model.predict(X_seq, verbose=0)
        
        # Inverse transform
        preds = self.scaler_y.inverse_transform(preds).flatten()
        
        # Pad with NaNs for the initial lookback period
        full_preds = np.full(len(X), np.nan)
        full_preds[self.lookback-1:] = preds
        
        return full_preds
    
    def feature_importance(self):
        """
        Get feature importance for LSTM model.
        
        LSTM models don't have a direct feature importance measure,
        so we use a permutation-based approach.
        
        Returns
        -------
        DataFrame or None
            Feature importance if available
        """
        # LSTM doesn't have a direct feature importance measure
        # We could implement a permutation-based approach,
        # but it's computationally expensive
        return None 