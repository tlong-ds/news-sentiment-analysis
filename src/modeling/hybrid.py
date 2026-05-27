"""Hybrid forecasting utilities for GARCH residuals plus sentiment-LSTM."""

from __future__ import annotations

# ruff: noqa: E402

import os
import torch
import logging

logger = logging.getLogger(__name__)

# Prevent OpenMP deadlocks on macOS/Unix by limiting thread pools
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)

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
    c: float = 0.0
    phi: float = 0.0
    gamma: float = 0.0


@dataclass(frozen=True)
class HybridForecastResult:
    feature_columns: list[str]
    sequence_length: int
    train_rows: int
    validation_rows: int
    test_rows: int


def fit_garch11_baseline(
    returns: Iterable[float], scale: float = 100.0
) -> GarchFitResult:
    """Estimate a Gaussian AR(1)-GARCH(1,1) model with constrained MLE."""
    series = pd.Series(np.asarray(list(returns), dtype=float)).dropna()
    if len(series) < 30:
        raise ValueError(
            "GARCH baseline needs at least 30 non-null return observations."
        )

    y = (series.to_numpy(copy=True)) * scale
    sample_var = float(np.var(y, ddof=1))

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
        c, phi, omega, alpha, beta = theta
        if (
            omega <= 0
            or alpha < 0
            or beta < 0
            or alpha + beta >= 0.999
            or abs(phi) >= 1.0
        ):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        return float(c), float(phi), float(omega), float(alpha), float(beta)

    def neg_loglike(theta: np.ndarray) -> float:
        c, phi, omega, alpha, beta = unpack(theta)
        if np.isnan(omega):
            return 1e12

        e = np.empty_like(y)
        e[0] = y[0] - c
        for idx in range(1, len(y)):
            e[idx] = y[idx] - c - phi * y[idx - 1]

        sigma2 = np.empty_like(y)
        sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
        for idx in range(1, len(y)):
            sigma2[idx] = omega + alpha * e[idx - 1] ** 2 + beta * sigma2[idx - 1]
            if sigma2[idx] <= 0 or not np.isfinite(sigma2[idx]):
                return 1e12

        ll = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (e**2) / sigma2)
        return float(-np.sum(ll))

    mean_y = float(np.mean(y))
    if len(y) > 1:
        phi_init = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        if not np.isfinite(phi_init) or abs(phi_init) >= 0.9:
            phi_init = 0.05
    else:
        phi_init = 0.05

    initial = np.array([mean_y, phi_init, sample_var * 0.05, 0.08, 0.9], dtype=float)
    bounds = [(None, None), (-0.999, 0.999), (1e-9, None), (1e-9, None), (1e-9, None)]
    constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[3] - x[4]}]
    result = minimize(
        neg_loglike,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"GARCH optimization failed: {result.message}")

    c, phi, omega, alpha, beta = unpack(result.x)

    e = np.empty_like(y)
    e[0] = y[0] - c
    for idx in range(1, len(y)):
        e[idx] = y[idx] - c - phi * y[idx - 1]

    sigma2 = np.empty_like(y)
    sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
    for idx in range(1, len(y)):
        sigma2[idx] = omega + alpha * e[idx - 1] ** 2 + beta * sigma2[idx - 1]

    forecast = np.empty_like(sigma2)
    forecast[:-1] = omega + alpha * e[:-1] ** 2 + beta * sigma2[:-1]
    forecast[-1] = omega + (alpha + beta) * sigma2[-1]
    standardized = e / np.sqrt(sigma2)

    return GarchFitResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_variance=sigma2 / (scale**2),
        variance_forecast=forecast / (scale**2),
        standardized_residuals=standardized,
        loss=float(result.fun),
        c=c,
        phi=phi,
    )


def fit_garchx11_baseline(
    returns: Iterable[float],
    exog: Iterable[float],
    scale: float = 100.0,
) -> GarchFitResult:
    """Estimate an AR(1)-GARCH-X(1,1) model with an exogenous variable in the variance equation."""
    series = pd.Series(np.asarray(list(returns), dtype=float)).dropna()
    exog_series = pd.Series(np.asarray(list(exog), dtype=float)).loc[series.index]
    if len(series) < 30:
        raise ValueError("GARCH-X baseline needs at least 30 observations.")

    y = (series.to_numpy(copy=True)) * scale
    x_exog = exog_series.to_numpy(copy=True)
    sample_var = float(np.var(y, ddof=1))

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float, float]:
        c, phi, omega, alpha, beta, gamma = theta
        if (
            omega <= 0
            or alpha < 0
            or beta < 0
            or alpha + beta >= 0.999
            or abs(phi) >= 1.0
        ):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        return (
            float(c),
            float(phi),
            float(omega),
            float(alpha),
            float(beta),
            float(gamma),
        )

    def neg_loglike(theta: np.ndarray) -> float:
        c, phi, omega, alpha, beta, gamma = unpack(theta)
        if np.isnan(omega):
            return 1e12

        e = np.empty_like(y)
        e[0] = y[0] - c
        for idx in range(1, len(y)):
            e[idx] = y[idx] - c - phi * y[idx - 1]

        sigma2 = np.empty_like(y)
        sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
        for idx in range(1, len(y)):
            sigma2[idx] = (
                omega
                + alpha * e[idx - 1] ** 2
                + beta * sigma2[idx - 1]
                + gamma * x_exog[idx - 1]
            )
            if sigma2[idx] <= 0 or not np.isfinite(sigma2[idx]):
                return 1e12

        ll = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (e**2) / sigma2)
        return float(-np.sum(ll))

    mean_y = float(np.mean(y))
    if len(y) > 1:
        phi_init = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        if not np.isfinite(phi_init) or abs(phi_init) >= 0.9:
            phi_init = 0.05
    else:
        phi_init = 0.05

    initial = np.array(
        [mean_y, phi_init, sample_var * 0.05, 0.08, 0.9, 0.0], dtype=float
    )
    bounds = [
        (None, None),
        (-0.999, 0.999),
        (1e-9, None),
        (1e-9, None),
        (1e-9, None),
        (None, None),
    ]
    constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[3] - x[4]}]
    result = minimize(
        neg_loglike,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"GARCH-X optimization failed: {result.message}")

    c, phi, omega, alpha, beta, gamma = unpack(result.x)

    e = np.empty_like(y)
    e[0] = y[0] - c
    for idx in range(1, len(y)):
        e[idx] = y[idx] - c - phi * y[idx - 1]

    sigma2 = np.empty_like(y)
    sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
    for idx in range(1, len(y)):
        sigma2[idx] = (
            omega
            + alpha * e[idx - 1] ** 2
            + beta * sigma2[idx - 1]
            + gamma * x_exog[idx - 1]
        )

    forecast = np.empty_like(sigma2)
    forecast[:-1] = (
        omega + alpha * e[:-1] ** 2 + beta * sigma2[:-1] + gamma * x_exog[:-1]
    )
    forecast[-1] = omega + (alpha + beta) * sigma2[-1] + gamma * x_exog[-1]
    standardized = e / np.sqrt(sigma2)

    return GarchFitResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_variance=sigma2 / (scale**2),
        variance_forecast=forecast / (scale**2),
        standardized_residuals=standardized,
        loss=float(result.fun),
        c=c,
        phi=phi,
        gamma=gamma,
    )


def fit_expanding_garch(
    returns: Iterable[float],
    train_len: int,
    reestimate_freq: int = 21,
    scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform expanding-window AR(1)-GARCH(1,1) volatility forecasting without look-ahead leakage.

    Re-estimates parameters c, phi, omega, alpha, beta every reestimate_freq steps.
    """
    series = pd.Series(np.asarray(list(returns), dtype=float)).dropna()
    y = series.to_numpy(copy=True) * scale
    n = len(y)

    sigma2 = np.empty(n)
    forecast = np.empty(n)

    garch_init = fit_garch11_baseline(series.iloc[:train_len], scale=scale)
    c = garch_init.c
    phi = garch_init.phi
    omega = garch_init.omega
    alpha = garch_init.alpha
    beta = garch_init.beta

    e = np.empty(n)
    e[0] = y[0] - c
    for idx in range(1, n):
        e[idx] = y[idx] - c - phi * y[idx - 1]

    sample_var = float(np.var(y[:train_len], ddof=1))
    sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
    forecast[0] = omega + alpha * e[0] ** 2 + beta * sigma2[0]

    for idx in range(1, n):
        if idx >= train_len and (idx - train_len) % reestimate_freq == 0:
            try:
                garch_temp = fit_garch11_baseline(series.iloc[:idx], scale=scale)
                c = garch_temp.c
                phi = garch_temp.phi
                omega = garch_temp.omega
                alpha = garch_temp.alpha
                beta = garch_temp.beta

                # Recalculate residuals up to idx using new parameters
                e[0] = y[0] - c
                for j in range(1, idx + 1):
                    e[j] = y[j] - c - phi * y[j - 1]
            except Exception:
                pass

        sigma2[idx] = omega + alpha * e[idx - 1] ** 2 + beta * sigma2[idx - 1]
        if idx < n - 1:
            forecast[idx] = omega + alpha * e[idx] ** 2 + beta * sigma2[idx]
        else:
            forecast[idx] = omega + (alpha + beta) * sigma2[idx]

    conditional_vol = np.sqrt(sigma2) / scale
    forecast_vol = np.sqrt(forecast) / scale
    std_resid = e / np.sqrt(sigma2)
    return conditional_vol, forecast_vol, std_resid


def add_garch_features(
    model_df: pd.DataFrame,
    *,
    return_column: str = "log_return",
    target_column: str = "target_next_vol",
    train_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Append GARCH variance and volatility forecasts and residual targets to the model frame.

    If train_end is specified, fits parameters c, phi, omega, alpha, beta, and training
    variance using only returns up to train_end to prevent look-ahead leakage,
    then projects GARCH variance forecasts and standardized residuals over
    the entire dataset using those fitted parameters.
    """
    df = model_df.copy()
    valid_index = df[df[return_column].notna()].index
    series = df.loc[valid_index, return_column]

    scale = 100.0
    y = series.to_numpy(copy=True) * scale

    if train_end is not None:
        # Identify the training subset using date comparison
        train_mask = pd.to_datetime(df.loc[valid_index, "date"]) <= pd.Timestamp(
            train_end
        )
        train_returns = series[train_mask]

        # Fit GARCH baseline on training return series only
        garch_train = fit_garch11_baseline(train_returns, scale=scale)
        c = garch_train.c
        phi = garch_train.phi
        omega = garch_train.omega
        alpha = garch_train.alpha
        beta = garch_train.beta

        # Project over the entire series using fitted parameters
        sample_var = float(np.var(train_returns.to_numpy(copy=True) * scale, ddof=1))

        e = np.empty_like(y)
        e[0] = y[0] - c
        for idx in range(1, len(y)):
            e[idx] = y[idx] - c - phi * y[idx - 1]

        sigma2 = np.empty_like(y)
        sigma2[0] = max(sample_var, omega / max(1e-6, 1.0 - alpha - beta))
        for idx in range(1, len(y)):
            sigma2[idx] = omega + alpha * e[idx - 1] ** 2 + beta * sigma2[idx - 1]

        forecast = np.empty_like(sigma2)
        forecast[:-1] = omega + alpha * e[:-1] ** 2 + beta * sigma2[:-1]
        forecast[-1] = omega + (alpha + beta) * sigma2[-1]
        standardized = e / np.sqrt(sigma2)

        # Convert variance back to volatility scale
        fitted_vol = np.sqrt(sigma2) / scale
        forecast_vol = np.sqrt(forecast) / scale
        fitted_var = sigma2 / (scale**2)
        forecast_var = forecast / (scale**2)
    else:
        # Fallback: fit on the full sample (original behavior)
        garch = fit_garch11_baseline(series, scale=scale)
        fitted_vol = np.sqrt(garch.conditional_variance)
        forecast_vol = np.sqrt(garch.variance_forecast)
        fitted_var = garch.conditional_variance
        forecast_var = garch.variance_forecast
        standardized = garch.standardized_residuals

    fitted = pd.Series(index=df.index, dtype=float)
    forecast_series = pd.Series(index=df.index, dtype=float)
    fitted_var_series = pd.Series(index=df.index, dtype=float)
    forecast_var_series = pd.Series(index=df.index, dtype=float)
    zscore = pd.Series(index=df.index, dtype=float)

    fitted.loc[valid_index] = fitted_vol
    forecast_series.loc[valid_index] = forecast_vol
    fitted_var_series.loc[valid_index] = fitted_var
    forecast_var_series.loc[valid_index] = forecast_var
    zscore.loc[valid_index] = standardized

    df["garch_conditional_vol"] = fitted
    df["garch_forecast_vol"] = forecast_series
    df["garch_conditional_var"] = fitted_var_series
    df["garch_forecast_var"] = forecast_var_series
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

    for end_idx in range(sequence_length - 1, len(usable)):
        window = usable.iloc[end_idx - sequence_length + 1 : end_idx + 1]
        target_row = usable.iloc[end_idx]
        x_values.append(window[feature_columns].to_numpy(dtype=float))
        y_values.append(float(target_row[target_column]))
        anchor_dates.append(pd.Timestamp(target_row["date"]))
        baseline_values.append(float(target_row.get("garch_forecast_vol", np.nan)))
        realized_values.append(float(target_row.get("target_next_vol", np.nan)))

    x = np.asarray(x_values, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    anchor_dates_arr = np.asarray(anchor_dates)
    baseline_arr = np.asarray(baseline_values, dtype=np.float32).reshape(-1, 1)
    realized_arr = np.asarray(realized_values, dtype=np.float32).reshape(-1, 1)

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


class ResidualLSTM(torch.nn.Module):
    """PyTorch equivalent of the TensorFlow Keras Sequential model for residual forecasting."""

    def __init__(self, input_dim: int, lstm_units: int = 32):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.fc1 = torch.nn.Linear(lstm_units, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Get the output of the last sequence step
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class PyTorchHistory:
    """Wrapper to mimic Keras history.history dict behavior for metrics logging."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "mae": [],
            "val_mae": [],
        }


def train_lstm_residual_model(
    sequences: dict[str, np.ndarray],
    *,
    epochs: int = 30,
    batch_size: int = 32,
    lstm_units: int = 32,
) -> tuple[torch.nn.Module, PyTorchHistory]:
    """Train the residual-correction LSTM using PyTorch."""
    import copy

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info("Training LSTM residual model on device: %s", device)

    x_train_np = sequences["x_train"]
    y_train_np = sequences["y_train"]
    input_dim = x_train_np.shape[-1]

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)

    has_validation = "x_val" in sequences and len(sequences["x_val"]) > 0
    if has_validation:
        x_val = torch.tensor(sequences["x_val"], dtype=torch.float32).to(device)
        y_val = torch.tensor(sequences["y_val"], dtype=torch.float32).to(device)

    model = ResidualLSTM(input_dim=input_dim, lstm_units=lstm_units).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    history = PyTorchHistory()
    best_loss = float("inf")
    best_weights = None
    patience = 5
    patience_counter = 0

    # Build simple DataLoaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            train_mae += torch.abs(outputs - batch_y).sum().item()

        train_loss /= len(x_train)
        train_mae /= len(x_train)

        history.history["loss"].append(train_loss)
        history.history["mae"].append(train_mae)

        log_msg = (
            f"Epoch {epoch}/{epochs} - loss: {train_loss:.6f} - mae: {train_mae:.6f}"
        )

        if has_validation:
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_mae = torch.abs(val_outputs - y_val).mean().item()

            history.history["val_loss"].append(val_loss)
            history.history["val_mae"].append(val_mae)
            log_msg += f" - val_loss: {val_loss:.6f} - val_mae: {val_mae:.6f}"

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break
        else:
            # If no validation set, just keep the latest weights
            best_weights = copy.deepcopy(model.state_dict())

        logger.info(log_msg)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
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
    clean_resid = result.standardized_residuals[
        ~np.isnan(result.standardized_residuals)
    ]

    lb_df = acorr_ljungbox(clean_resid**2, lags=[5, 10], return_df=True)
    p_val_5 = float(lb_df.loc[5, "lb_pvalue"])
    p_val_10 = float(lb_df.loc[10, "lb_pvalue"])

    # No remaining ARCH if p-value > 0.05 (fail to reject null of no autocorrelation)
    no_remaining_arch = bool(p_val_5 > 0.05 and p_val_10 > 0.05)

    return {
        "alpha_plus_beta": float(alpha_plus_beta),
        "stationary": stationary,
        "ljung_box_pvalue_lag5": p_val_5,
        "ljung_box_pvalue_lag10": p_val_10,
        "no_remaining_arch_effects": no_remaining_arch,
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
    var_d = (
        gamma[0]
        + 2.0
        * np.sum(
            [
                ((1.0 - (lag / (max_lag + 1))) * gamma[lag])
                for lag in range(1, max_lag + 1)
            ]
        )
    ) / n

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
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "actual": actual.reshape(-1),
            "baseline": pred_baseline.reshape(-1),
            "hybrid": pred_hybrid.reshape(-1),
            "sentiment": sentiment.reshape(-1),
        }
    )

    df["year"] = df["date"].dt.year
    df["baseline_error"] = (df["actual"] - df["baseline"]).abs()
    df["hybrid_error"] = (df["actual"] - df["hybrid"]).abs()

    analysis = {}

    # 1. Year-by-year splits
    for yr in df["year"].unique():
        yr_df = df[df["year"] == yr]
        analysis[f"year_{yr}"] = {
            "size": len(yr_df),
            "baseline_rmse": float(
                np.sqrt(np.mean((yr_df["actual"] - yr_df["baseline"]) ** 2))
            ),
            "hybrid_rmse": float(
                np.sqrt(np.mean((yr_df["actual"] - yr_df["hybrid"]) ** 2))
            ),
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
                "baseline_rmse": float(
                    np.sqrt(np.mean((sub_df["actual"] - sub_df["baseline"]) ** 2))
                ),
                "hybrid_rmse": float(
                    np.sqrt(np.mean((sub_df["actual"] - sub_df["hybrid"]) ** 2))
                ),
                "baseline_mae": float(np.mean(sub_df["baseline_error"])),
                "hybrid_mae": float(np.mean(sub_df["hybrid_error"])),
            }

    # 3. Sentiment Asymmetry (Negative vs Positive sentiment days)
    neg_df = df[df["sentiment"] < -0.05]
    pos_df = df[df["sentiment"] > 0.05]

    for label, sub_df in [
        ("negative_sentiment_days", neg_df),
        ("positive_sentiment_days", pos_df),
    ]:
        if len(sub_df) > 0:
            analysis[label] = {
                "size": len(sub_df),
                "baseline_rmse": float(
                    np.sqrt(np.mean((sub_df["actual"] - sub_df["baseline"]) ** 2))
                ),
                "hybrid_rmse": float(
                    np.sqrt(np.mean((sub_df["actual"] - sub_df["hybrid"]) ** 2))
                ),
                "baseline_mae": float(np.mean(sub_df["baseline_error"])),
                "hybrid_mae": float(np.mean(sub_df["hybrid_error"])),
            }

    return analysis
