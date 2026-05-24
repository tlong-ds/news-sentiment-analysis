"""Hybrid forecasting utilities for GARCH residuals plus sentiment-LSTM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class GarchFitResult:
    omega: float
    alpha: float
    beta: float
    conditional_variance: np.ndarray
    variance_forecast: np.ndarray
    standardized_residuals: np.ndarray
    loss: float


@dataclass(frozen=True)
class HybridForecastResult:
    feature_columns: list[str]
    sequence_length: int
    train_rows: int
    validation_rows: int
    test_rows: int


def fit_garch11_baseline(returns: Iterable[float], scale: float = 100.0) -> GarchFitResult:
    """Estimate a Gaussian GARCH(1,1) model with constrained MLE."""
    series = pd.Series(np.asarray(list(returns), dtype=float)).dropna()
    if len(series) < 30:
        raise ValueError("GARCH baseline needs at least 30 non-null return observations.")

    y = (series.to_numpy(copy=True)) * scale
    sample_var = float(np.var(y, ddof=1))

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return np.nan, np.nan, np.nan
        return float(omega), float(alpha), float(beta)

    def neg_loglike(theta: np.ndarray) -> float:
        omega, alpha, beta = unpack(theta)
        if np.isnan(omega):
            return 1e12

        sigma2 = np.empty_like(y)
        sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
        for idx in range(1, len(y)):
            sigma2[idx] = omega + alpha * y[idx - 1] ** 2 + beta * sigma2[idx - 1]
            if sigma2[idx] <= 0 or not np.isfinite(sigma2[idx]):
                return 1e12

        ll = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (y**2) / sigma2)
        return float(-np.sum(ll))

    initial = np.array([sample_var * 0.05, 0.08, 0.9], dtype=float)
    bounds = [(1e-9, None), (1e-9, 0.999), (1e-9, 0.999)]
    constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]}]
    result = minimize(
        neg_loglike,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"GARCH optimization failed: {result.message}")

    omega, alpha, beta = unpack(result.x)
    sigma2 = np.empty_like(y)
    sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
    for idx in range(1, len(y)):
        sigma2[idx] = omega + alpha * y[idx - 1] ** 2 + beta * sigma2[idx - 1]

    forecast = np.empty_like(sigma2)
    forecast[:-1] = omega + alpha * y[:-1] ** 2 + beta * sigma2[:-1]
    forecast[-1] = omega + (alpha + beta) * sigma2[-1]
    standardized = y / np.sqrt(sigma2)

    return GarchFitResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_variance=sigma2 / (scale**2),
        variance_forecast=forecast / (scale**2),
        standardized_residuals=standardized,
        loss=float(result.fun),
    )


def add_garch_features(
    model_df: pd.DataFrame,
    *,
    return_column: str = "log_return",
    target_column: str = "target_next_vol",
) -> pd.DataFrame:
    """Append GARCH variance forecasts and residual targets to the model frame."""
    garch = fit_garch11_baseline(model_df[return_column])
    df = model_df.copy()
    fitted = pd.Series(index=df.index, dtype=float)
    forecast = pd.Series(index=df.index, dtype=float)
    zscore = pd.Series(index=df.index, dtype=float)

    valid_index = df[df[return_column].notna()].index
    fitted.loc[valid_index] = np.sqrt(garch.conditional_variance)
    forecast.loc[valid_index] = np.sqrt(garch.variance_forecast)
    zscore.loc[valid_index] = garch.standardized_residuals

    df["garch_conditional_vol"] = fitted
    df["garch_forecast_vol"] = forecast
    df["garch_std_resid"] = zscore
    df["hybrid_residual_target"] = df[target_column] - df["garch_forecast_vol"]
    return df


def build_lstm_sequences(
    model_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int = 10,
    split_dates: tuple[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], HybridForecastResult]:
    """Transform a daily frame into rolling sequences for the LSTM stage."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")

    df = model_df.copy().sort_values("date").reset_index(drop=True)
    needed = ["date", target_column, *feature_columns]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for LSTM dataset: {missing}")

    usable = df.dropna(subset=needed).reset_index(drop=True)
    if len(usable) <= sequence_length:
        raise ValueError("Not enough rows to build LSTM sequences.")

    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    anchor_dates: list[pd.Timestamp] = []
    baseline_values: list[float] = []
    realized_values: list[float] = []

    for end_idx in range(sequence_length, len(usable)):
        window = usable.iloc[end_idx - sequence_length : end_idx]
        target_row = usable.iloc[end_idx]
        x_values.append(window[feature_columns].to_numpy(dtype=float))
        y_values.append(float(target_row[target_column]))
        anchor_dates.append(pd.Timestamp(target_row["date"]))
        baseline_values.append(float(target_row.get("garch_forecast_vol", np.nan)))
        realized_values.append(float(target_row.get("target_next_vol", np.nan)))

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float).reshape(-1, 1)
    anchor_dates_arr = np.asarray(anchor_dates)
    baseline_arr = np.asarray(baseline_values, dtype=float).reshape(-1, 1)
    realized_arr = np.asarray(realized_values, dtype=float).reshape(-1, 1)

    if split_dates is None:
        train_cut = int(len(x) * 0.7)
        val_cut = int(len(x) * 0.85)
        masks = {
            "train": np.arange(len(x)) < train_cut,
            "val": (np.arange(len(x)) >= train_cut) & (np.arange(len(x)) < val_cut),
            "test": np.arange(len(x)) >= val_cut,
        }
    else:
        train_end, val_end = (pd.Timestamp(part) for part in split_dates)
        masks = {
            "train": anchor_dates_arr <= train_end,
            "val": (anchor_dates_arr > train_end) & (anchor_dates_arr <= val_end),
            "test": anchor_dates_arr > val_end,
        }

    bundle: dict[str, np.ndarray] = {}
    for split, mask in masks.items():
        bundle[f"x_{split}"] = x[mask]
        bundle[f"y_{split}"] = y[mask]
        bundle[f"dates_{split}"] = anchor_dates_arr[mask]
        bundle[f"baseline_{split}"] = baseline_arr[mask]
        bundle[f"realized_{split}"] = realized_arr[mask]

    meta = HybridForecastResult(
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        train_rows=int(masks["train"].sum()),
        validation_rows=int(masks["val"].sum()),
        test_rows=int(masks["test"].sum()),
    )
    return bundle, meta


def train_lstm_residual_model(
    sequences: dict[str, np.ndarray],
    *,
    epochs: int = 30,
    batch_size: int = 32,
    lstm_units: int = 32,
):
    """Train the residual-correction LSTM if TensorFlow is available."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required for the LSTM stage. Install it from "
            "requirements.txt before training the hybrid model."
        ) from exc

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=sequences["x_train"].shape[1:]),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        sequences["x_train"],
        sequences["y_train"],
        validation_data=(sequences["x_val"], sequences["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    return model, history


def evaluate_forecasts(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute standard volatility-forecast error metrics."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length.")

    error = actual - predicted
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    mape = float(np.mean(np.abs(error) / np.clip(np.abs(actual), 1e-8, None)))
    return {"rmse": rmse, "mae": mae, "mape": mape}


def validate_garch_fit(result: GarchFitResult) -> dict[str, float | bool]:
    """Perform stationarity and residual diagnostics on the GARCH baseline."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    alpha_plus_beta = result.alpha + result.beta
    stationary = bool(alpha_plus_beta < 1.0)
    
    # Ljung-Box test on squared standardized residuals (checks for remaining ARCH effects)
    clean_resid = result.standardized_residuals[~np.isnan(result.standardized_residuals)]
    
    lb_df = acorr_ljungbox(clean_resid ** 2, lags=[5, 10], return_df=True)
    p_val_5 = float(lb_df.loc[5, "lb_pvalue"])
    p_val_10 = float(lb_df.loc[10, "lb_pvalue"])
    
    # No remaining ARCH if p-value > 0.05 (fail to reject null of no autocorrelation)
    no_remaining_arch = bool(p_val_5 > 0.05 and p_val_10 > 0.05)
    
    return {
        "alpha_plus_beta": float(alpha_plus_beta),
        "stationary": stationary,
        "ljung_box_pvalue_lag5": p_val_5,
        "ljung_box_pvalue_lag10": p_val_10,
        "no_remaining_arch_effects": no_remaining_arch
    }


def diebold_mariano_test(
    actual: np.ndarray,
    pred_baseline: np.ndarray,
    pred_hybrid: np.ndarray,
    loss_type: str = "square",
    max_lag: int = 1,
) -> tuple[float, float]:
    """Perform the Diebold-Mariano test for forecast accuracy with Newey-West variance."""
    from scipy.stats import norm

    actual = np.asarray(actual, dtype=float).reshape(-1)
    pred1 = np.asarray(pred_baseline, dtype=float).reshape(-1)
    pred2 = np.asarray(pred_hybrid, dtype=float).reshape(-1)
    
    if len(actual) != len(pred1) or len(actual) != len(pred2):
        raise ValueError("All inputs must have the same length.")
        
    if loss_type == "square":
        d = (actual - pred1) ** 2 - (actual - pred2) ** 2
    elif loss_type == "absolute":
        d = np.abs(actual - pred1) - np.abs(actual - pred2)
    elif loss_type == "qlike":
        # QLIKE: ln(pred) + actual/pred
        d = (np.log(pred1) + actual / np.clip(pred1, 1e-8, None)) - (
            np.log(pred2) + actual / np.clip(pred2, 1e-8, None)
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    mean_d = np.mean(d)
    n = len(d)
    
    # Sample autocovariances up to max_lag
    gamma = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            gamma[lag] = np.var(d, ddof=0)
        else:
            gamma[lag] = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
            
    # Newey-West variance estimator
    var_d = (gamma[0] + 2.0 * np.sum(
        [((1.0 - (lag / (max_lag + 1))) * gamma[lag]) for lag in range(1, max_lag + 1)]
    )) / n
    
    if var_d <= 0:
        return 0.0, 1.0
        
    dm_stat = float(mean_d / np.sqrt(var_d))
    # Two-sided p-value
    p_value = float(2.0 * (1.0 - norm.cdf(np.abs(dm_stat))))
    return dm_stat, p_value


def analyze_forecast_subperiods(
    actual: np.ndarray,
    pred_baseline: np.ndarray,
    pred_hybrid: np.ndarray,
    dates: np.ndarray,
    sentiment: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Segment test forecast performance by subperiod regimes and sentiment conditions."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "actual": actual.reshape(-1),
        "baseline": pred_baseline.reshape(-1),
        "hybrid": pred_hybrid.reshape(-1),
        "sentiment": sentiment.reshape(-1),
    })
    
    df["year"] = df["date"].dt.year
    df["baseline_error"] = (df["actual"] - df["baseline"]).abs()
    df["hybrid_error"] = (df["actual"] - df["hybrid"]).abs()
    
    analysis = {}
    
    # 1. Year-by-year splits
    for yr in df["year"].unique():
        yr_df = df[df["year"] == yr]
        analysis[f"year_{yr}"] = {
            "size": len(yr_df),
            "baseline_rmse": float(np.sqrt(np.mean((yr_df["actual"] - yr_df["baseline"])**2))),
            "hybrid_rmse": float(np.sqrt(np.mean((yr_df["actual"] - yr_df["hybrid"])**2))),
            "baseline_mae": float(np.mean(yr_df["baseline_error"])),
            "hybrid_mae": float(np.mean(yr_df["hybrid_error"])),
        }
        
    # 2. Volatility Regimes (Shock vs Calm)
    median_vol = df["actual"].median()
    shock_df = df[df["actual"] > median_vol]
    calm_df = df[df["actual"] <= median_vol]
    
    for label, sub_df in [("shock_regime", shock_df), ("calm_regime", calm_df)]:
        if len(sub_df) > 0:
            analysis[label] = {
                "size": len(sub_df),
                "baseline_rmse": float(np.sqrt(np.mean((sub_df["actual"] - sub_df["baseline"])**2))),
                "hybrid_rmse": float(np.sqrt(np.mean((sub_df["actual"] - sub_df["hybrid"])**2))),
                "baseline_mae": float(np.mean(sub_df["baseline_error"])),
                "hybrid_mae": float(np.mean(sub_df["hybrid_error"])),
            }
            
    # 3. Sentiment Asymmetry (Negative vs Positive sentiment days)
    neg_df = df[df["sentiment"] < -0.05]
    pos_df = df[df["sentiment"] > 0.05]
    
    for label, sub_df in [("negative_sentiment_days", neg_df), ("positive_sentiment_days", pos_df)]:
        if len(sub_df) > 0:
            analysis[label] = {
                "size": len(sub_df),
                "baseline_rmse": float(np.sqrt(np.mean((sub_df["actual"] - sub_df["baseline"])**2))),
                "hybrid_rmse": float(np.sqrt(np.mean((sub_df["actual"] - sub_df["hybrid"])**2))),
                "baseline_mae": float(np.mean(sub_df["baseline_error"])),
                "hybrid_mae": float(np.mean(sub_df["hybrid_error"])),
            }
            
    return analysis

