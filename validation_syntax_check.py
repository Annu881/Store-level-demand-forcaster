import numpy as np
import pandas as pd


def recursive_xgb_validation(xgb_model, val_rec, hist, feature_names):
    """Evaluate d_1886-d_1913 recursively for thresholds 0.0, 0.5, and 1.0."""
    start_day, end_day = 1886, 1913
    validation_days = [f"d_{day}" for day in range(start_day, end_day + 1)]
    history_days = [f"d_{day}" for day in range(start_day - 56, start_day)]
    thresholds = (0.0, 0.5, 1.0)
    dynamic_features = {
        "lag_1", "lag_7", "lag_14", "lag_28", "lag_56",
        "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
        "zero_ratio_7", "zero_ratio_28", "days_since_last_sale",
        "no_previous_sale",
    }

    feature_names = list(feature_names)
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("FEATURES contains duplicate feature names.")
    if not dynamic_features.issubset(feature_names):
        missing = sorted(dynamic_features.difference(feature_names))
        raise ValueError(f"FEATURES is missing dynamic features: {missing}")

    required_val = {"id", "d", "sales"}.union(feature_names)
    missing_val = sorted(required_val.difference(val_rec.columns))
    if missing_val:
        raise ValueError(f"Validation data is missing columns: {missing_val}")
    if val_rec.duplicated(["id", "d"]).any():
        raise ValueError("Validation data has duplicate (id, d) rows.")
    if set(val_rec["d"].unique()) != set(validation_days):
        raise ValueError("Validation data must contain exactly d_1886 through d_1913.")

    ids_by_day = {
        day_name: set(val_rec.loc[val_rec["d"] == day_name, "id"])
        for day_name in validation_days
    }
    series_ids = val_rec.loc[val_rec["d"] == validation_days[0], "id"].tolist()
    if not series_ids or any(ids != set(series_ids) for ids in ids_by_day.values()):
        raise ValueError("Every validation day must contain the same non-empty series set.")

    required_hist = {"id", "d", "sales"}
    missing_hist = sorted(required_hist.difference(hist.columns))
    if missing_hist:
        raise ValueError(f"Historical data is missing columns: {missing_hist}")
    history_slice = hist.loc[
        hist["d"].isin(history_days), ["id", "d", "sales"]
    ].copy()
    if history_slice.duplicated(["id", "d"]).any():
        raise ValueError("Historical data has duplicate (id, d) rows.")

    history_wide = history_slice.pivot(index="id", columns="d", values="sales")
    missing_history_days = sorted(set(history_days).difference(history_wide.columns))
    if missing_history_days:
        raise ValueError(f"Historical data is missing required days: {missing_history_days}")
    history_wide = history_wide.reindex(index=series_ids, columns=history_days)
    if history_wide.isna().any().any():
        raise ValueError("Historical sales have missing values for validation series/days.")
    history_base = history_wide.copy()

    first_day = (
        val_rec.loc[val_rec["d"] == validation_days[0]]
        .set_index("id")
        .reindex(series_ids)
    )
    initial_days_since = pd.to_numeric(
        first_day["days_since_last_sale"], errors="raise"
    )
    initial_no_previous = pd.to_numeric(
        first_day["no_previous_sale"], errors="raise"
    )
    if initial_days_since.isna().any() or initial_no_previous.isna().any():
        raise ValueError("The first validation day's sale-history features contain missing values.")
    if not initial_no_previous.isin([0, 1]).all():
        raise ValueError("no_previous_sale must contain only 0 or 1.")
    never_sold = initial_no_previous.eq(1)
    if not (initial_days_since.loc[never_sold] == -1).all():
        raise ValueError("Training sentinel mismatch: never-sold series must have days_since_last_sale=-1.")
    if not (initial_days_since.loc[~never_sold] >= 1).all():
        raise ValueError("Previously-sold series must have days_since_last_sale >= 1.")

    # Seed from d_1886's pre-day features, which summarize actual history only through d_1885.
    last_sale_day = pd.Series(np.nan, index=series_ids, dtype=np.float64)
    has_previous_sale = ~never_sold
    last_sale_day.loc[has_previous_sale] = (
        start_day - initial_days_since.loc[has_previous_sale].astype(np.float64)
    )

    summary_rows = []
    horizon_rows = []
    prediction_frames = []

    for threshold in thresholds:
        history = history_base.copy()
        last_sale = last_sale_day.copy()
        actual_by_day = []
        prediction_by_day = []
        prediction_rows = []
        predicted_sale_days = 0

        for horizon, day in enumerate(range(start_day, end_day + 1), start=1):
            day_name = f"d_{day}"
            day_df = (
                val_rec.loc[val_rec["d"] == day_name]
                .set_index("id")
                .reindex(series_ids)
                .reset_index()
                .copy()
            )
            ids = day_df["id"].tolist()
            recent_days = [f"d_{value}" for value in range(day - 56, day)]
            recent = history.loc[ids, recent_days].to_numpy(dtype=np.float32)

            last_sale_today = last_sale.loc[ids]
            has_sale = last_sale_today.notna().to_numpy()
            days_since = np.full(len(ids), -1.0, dtype=np.float32)
            last_sale_values = last_sale_today.to_numpy(dtype=np.float64)
            days_since[has_sale] = day - last_sale_values[has_sale]

            dynamic_values = {
                "lag_1": recent[:, -1],
                "lag_7": recent[:, -7],
                "lag_14": recent[:, -14],
                "lag_28": recent[:, -28],
                "lag_56": recent[:, 0],
                "roll_mean_7": recent[:, -7:].mean(axis=1),
                "roll_std_7": recent[:, -7:].std(axis=1, ddof=1),
                "roll_mean_28": recent[:, -28:].mean(axis=1),
                "roll_std_28": recent[:, -28:].std(axis=1, ddof=1),
                "zero_ratio_7": (recent[:, -7:] == 0).mean(axis=1),
                "zero_ratio_28": (recent[:, -28:] == 0).mean(axis=1),
                "days_since_last_sale": days_since,
                "no_previous_sale": (~has_sale).astype(np.int8),
            }
            for name, values in dynamic_values.items():
                day_df[name] = values

            actual = pd.to_numeric(day_df["sales"], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(actual).all():
                raise ValueError(f"Actual sales contain non-finite values on {day_name}.")
            X_day = day_df.loc[:, feature_names].to_numpy(dtype=np.float32)
            prediction = np.asarray(xgb_model.predict(X_day), dtype=np.float64).reshape(-1)
            if len(prediction) != len(ids) or not np.isfinite(prediction).all():
                raise ValueError(f"Model returned invalid predictions on {day_name}.")
            prediction = np.maximum(prediction, 0.0)

            # The same threshold mask is recorded and used to update recursive sale state.
            update_last_sale = prediction > threshold
            predicted_sale_days += int(update_last_sale.sum())
            prediction_error = prediction - actual
            horizon_rows.append({
                "Threshold": threshold,
                "Horizon": horizon,
                "Day": day_name,
                "MAE": float(np.mean(np.abs(prediction_error))),
            })
            prediction_rows.append(pd.DataFrame({
                "Threshold": np.full(len(ids), threshold, dtype=np.float32),
                "Horizon": np.full(len(ids), horizon, dtype=np.int8),
                "Day": day_name,
                "id": ids,
                "Actual": actual.astype(np.float32),
                "Prediction": prediction.astype(np.float32),
                "LastSaleUpdated": update_last_sale,
            }))
            actual_by_day.append(actual)
            prediction_by_day.append(prediction)

            # Feed predictions, never validation actuals, into the next day's history.
            history[day_name] = np.nan
            history.loc[ids, day_name] = prediction
            last_sale.loc[np.asarray(ids)[update_last_sale]] = day

        all_actual = np.concatenate(actual_by_day)
        all_prediction = np.concatenate(prediction_by_day)
        error = all_prediction - all_actual
        actual_total = float(np.abs(all_actual).sum())
        if actual_total == 0:
            raise ValueError("WAPE is undefined because validation actuals sum to zero.")
        summary_rows.append({
            "Threshold": threshold,
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(np.square(error)))),
            "WAPE_pct": float(100.0 * np.abs(error).sum() / actual_total),
            "PredictionBias": float(np.mean(error)),
            "PredictedSaleDays": predicted_sale_days,
        })
        prediction_frames.append(pd.concat(prediction_rows, ignore_index=True))

    summary = pd.DataFrame(summary_rows)
    horizon_mae = pd.DataFrame(horizon_rows)
    prediction_comparison = pd.concat(prediction_frames, ignore_index=True)
    prediction_comparison["id"] = prediction_comparison["id"].astype("category")
    prediction_comparison["Day"] = pd.Categorical(
        prediction_comparison["Day"], categories=validation_days, ordered=True
    )
    prediction_comparison["Threshold"] = pd.Categorical(
        prediction_comparison["Threshold"], categories=thresholds, ordered=True
    )
    return summary, horizon_mae, prediction_comparison
