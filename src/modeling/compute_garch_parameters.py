"""Fit baseline GARCH(1,1) on the training set and print estimated parameters."""

from src.modeling.dataset import load_or_build_model_frame
from src.modeling.hybrid import fit_garch11_baseline


def main():
    prices = "data/raw/prices_VN.csv"
    model_frame = "data/interim/modeling_ready.parquet"
    daily_news = "data/interim/daily_news_prices.parquet"

    # Load model frame
    model_df = load_or_build_model_frame(
        model_frame_path=model_frame,
        price_path=prices,
        daily_news_path=daily_news,
    )

    # Train split: up to 2021-12-31
    train_df = model_df[model_df["date"] <= "2021-12-31"]

    # Fit GARCH(1,1)
    garch_res = fit_garch11_baseline(train_df["log_return"])

    print("\n=== Fitted GARCH(1,1) Baseline Parameters on Training Set ===")
    print(f"omega (unscaled): {garch_res.omega}")
    # Note: the returns are scaled by 100 during fitting, so let's check the scaled parameters too
    print(
        f"omega (scaled): {garch_res.omega}"
    )  # wait, fit_garch11_baseline scales by 100 internally, and returns unscaled or scaled?
    # Let's check:
    # y = returns * scale (scale=100)
    # the returned conditional variance is divided by (scale**2).
    # wait! Let's see if the omega printed is scaled or not.
    # In fit_garch11_baseline, the returned conditional_variance is scaled_var / (scale**2), so it is unscaled.
    # But what about omega, alpha, beta?
    # The return object returns the raw omega, alpha, beta from the optimizer. Since optimizer is run on scaled returns y,
    # the omega is scaled (i.e. for returns in %). The alpha and beta are scale-invariant.
    # Let's print them:
    print(f"omega (for y in %): {garch_res.omega:.6f}")
    print(f"alpha: {garch_res.alpha:.6f}")
    print(f"beta: {garch_res.beta:.6f}")
    print(f"alpha + beta: {garch_res.alpha + garch_res.beta:.6f}")


if __name__ == "__main__":
    main()
