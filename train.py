"""
Demand forecast training pipeline.

Code order (อ่านง่ายและ debug ง่าย):
1) Data access / config loading
2) Feature engineering + forecasting helpers
3) Training orchestration
4) Output shaping + upload
"""

from botocore.utils import ClientError
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from joblib import Parallel, delayed
from sqlalchemy import text
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import poisson
import shutil
# import matplotlib.pyplot as plt
import boto3
from botocore.client import Config
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from demand_forecast.config import normalize_training_config
from demand_forecast.database_engine import db_manager
from demand_forecast.prep_demandforecast_data import prep_demandforecast_data


# ---------------------------------------------------------------------------
# 1) Data access / config loading
# ---------------------------------------------------------------------------
# Load environment variables from .env
load_dotenv()

def get_holiday_df() -> pd.DataFrame:   
    """Load holiday table via shared Supabase engine."""
    query = text("SELECT * FROM warehouse.holiday;")
    holiday_df = pd.read_sql(query, db_manager.supabase_engine)
    print(f"[DATA] Loaded holiday data: {holiday_df.shape}")
    return holiday_df


def get_oms_pandas_df() -> pd.DataFrame:
    """
    Build demand dataframe from prep_demandforecast_data pipeline.
    """
    df = prep_demandforecast_data()
    print(f"[DATA] Prepared data: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2) Feature engineering + forecasting helpers
# ---------------------------------------------------------------------------
# IMPACT DATE เป็นการตัดสินใจจากค่า CoV + rolling z-score

def extract_impact_dates_with_cv(
    join_outbound_df,
    top_skus_n=20,
    cv_bands: list[float] | None = None,
    zscore_rules: list[dict[str, float]] | None = None,
):
    cv_bands = cv_bands or [0.2, 0.6]
    if len(cv_bands) < 2:
        cv_bands = [0.2, 0.6]

    zscore_rules = zscore_rules or [
        {"window": 10, "threshold": 2.5},
        {"window": 7, "threshold": 2.0},
        {"window": 5, "threshold": 1.5},
    ]
    # Keep deterministic order by window desc.
    zscore_rules = sorted(
        zscore_rules,
        key=lambda x: int(x.get("window", 0)),
        reverse=True,
    )

    top_skus = join_outbound_df.groupby('Shop_SKU')['y'].sum().nlargest(top_skus_n).index
    all_impact_dates = []

    for sku in top_skus:
        sku_df = join_outbound_df[join_outbound_df['Shop_SKU'] == sku].sort_values('ds').copy()
        cv = sku_df['y'].std() / sku_df['y'].mean()

        if np.isnan(cv) or cv < float(cv_bands[0]):
            rule = zscore_rules[0]
        elif cv < float(cv_bands[1]):
            rule = zscore_rules[min(1, len(zscore_rules) - 1)]
        else:
            rule = zscore_rules[min(2, len(zscore_rules) - 1)]

        window = int(rule.get("window", 7))
        threshold = float(rule.get("threshold", 2.0))

        sku_df['rolling_mean'] = sku_df['y'].rolling(window=window).mean()
        sku_df['rolling_std'] = sku_df['y'].rolling(window=window).std()

        sku_df["rolling_std"] = sku_df["rolling_std"].replace(0, np.nan)
        sku_df['z_score'] = (sku_df['y'] - sku_df['rolling_mean']) / sku_df['rolling_std']
        sku_df['z_score'] = sku_df['z_score'].replace([np.inf, -np.inf], np.nan)

        impact_dates = sku_df.loc[sku_df['z_score'].abs() > threshold, 'ds'].tolist()
        all_impact_dates.extend(impact_dates)

    return list(pd.to_datetime(pd.Series(all_impact_dates).dropna().unique()))

def build_impact_holidays_df(impact_dates, config):
    if not impact_dates:
        return pd.DataFrame(columns=["holiday","ds","lower_window","upper_window"])
    return pd.DataFrame({
        "holiday": "impact_dates",
        "ds": pd.to_datetime(impact_dates),
        "lower_window": config["impact_dates"]["lower_window"],
        "upper_window": config["impact_dates"]["upper_window"]
    })

#3) ใช้เฉพาะกรณีมีDemand ช่วงforecast period > 30 days
def compute_bias_correction(join_outbound_df, forecast_df_all, min_demand_threshold=1000, min_positive_days=30):
    correction_map = {}

    for sku in forecast_df_all['Shop_SKU'].unique():
        actual_series = join_outbound_df[join_outbound_df['Shop_SKU'] == sku].set_index('ds')['y']
        forecast_series = forecast_df_all[forecast_df_all['Shop_SKU'] == sku].set_index('ds')['yhat']

        common_dates = actual_series.index.intersection(forecast_series.index)
        if len(common_dates) == 0:
            continue

        actual = actual_series.loc[common_dates]
        forecast = forecast_series.loc[common_dates]

        # ✅ กรอง SKU ที่ยอดรวมต่ำมาก
        if actual.sum() < min_demand_threshold:
            correction = 0
        else:
            residual = actual - forecast
            positive_residuals = residual[residual > 0]

            # ✅ ถ้ามี spike วันเดียว → ไม่ต้อง correction
            if len(positive_residuals) < min_positive_days:
                correction = 0
            else:
                correction = positive_residuals.median()

        correction_map[sku] = correction

    return correction_map

def apply_correction(forecast_df_all, correction_map):
    forecast_df_all['yhat_corrected'] = forecast_df_all.apply(
        lambda row: row['yhat'] + correction_map.get(row['Shop_SKU'], 0), axis=1
    ).round().astype(int)
    return forecast_df_all

#4) Lag features (config-driven)
def add_lag_features(
    sku_df: pd.DataFrame,
    lag_days: list[int] | None = None,
    rolling_mean_windows: list[int] | None = None,
):
    sku_df = sku_df.sort_values("ds").copy()
    lag_days = lag_days or [1, 7]
    rolling_mean_windows = rolling_mean_windows or [3]

    lag_feature_cols: list[str] = []
    for lag in lag_days:
        col = f"y_lag{int(lag)}"
        sku_df[col] = sku_df["y"].shift(int(lag))
        lag_feature_cols.append(col)

    for window in rolling_mean_windows:
        col = f"rolling_mean_{int(window)}"
        sku_df[col] = sku_df["y"].rolling(window=int(window)).mean().shift(1)
        lag_feature_cols.append(col)

    return sku_df, lag_feature_cols

#5) Normalize promotion input
def normalize_promotions(promo_json_list):
    """
    Input: list[dict] ตาม format ALL/INCLUDE
    Output: promo_master_df ระดับโปร (ยังไม่ explode รายวัน)
    """
    rows = []
    for p in promo_json_list:
        base = {
            "customer_code": p.get("customer_code"),
            "promotion_name": p.get("promotion_name"),
            "promotion_code": p.get("promotion_code"),
            "start_date": p.get("start_date"),
            "end_date": p.get("end_date"),
            "apply_mode": p.get("apply_mode", "ALL").upper(),
        }

        if base["apply_mode"] == "ALL": 
            disc = p.get("discount", {})
            rows.append({
                **base,
                "sku_code": None,  # ALL → เดี๋ยว expand ทีหลัง
                "discount_amount": disc.get("amount"),
                "discount_type": (disc.get("type") or "").lower(),  # fixed/percent
            })

        elif base["apply_mode"] == "INCLUDE":
            for s in p.get("include_sku", []):
                rows.append({
                    **base,
                    "sku_code": s.get("sku_code"),
                    "discount_amount": s.get("discount_amount"),
                    "discount_type": (s.get("discount_type") or "").lower(),
                })
        else:
            raise ValueError(f"Unknown apply_mode: {base['apply_mode']}")

    promo_master_df = pd.DataFrame(rows)
    return promo_master_df

#6) Build promo to be daily df
def build_promo_daily_df(promo_master_df, all_skus):
    """
    Output:
    ds, Shop_SKU, promo_flag, promo_discount_percent, promo_discount_fixed
    """
    print("[BUILD PROMO] using version WITH promo_flag")

    # ✅ guard clause (สำคัญมาก)
    if promo_master_df is None or promo_master_df.empty:
        return pd.DataFrame(
            columns=[
                "ds",
                "Shop_SKU",
                "promo_flag",
             #   "promo_discount_percent",
              #  "promo_discount_fixed",
            ]
        )

    df = promo_master_df.copy()

    # parse date
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    df = df.dropna(subset=["start_date", "end_date"])
    df = df[df["end_date"] >= df["start_date"]]

    expanded_rows = []

    for _, r in df.iterrows():
        if r["apply_mode"] == "ALL":
            target_skus = all_skus
        else:  # INCLUDE
            target_skus = [r["sku_code"]] if pd.notna(r["sku_code"]) else []

        for sku in target_skus:
            for d in pd.date_range(r["start_date"], r["end_date"], freq="D"):
                expanded_rows.append({
                    "ds": d,
                    "Shop_SKU": sku,
                    "promo_flag": 1,
                 #   "promo_discount_percent": (
                  #      r["discount_amount"] if r["discount_type"] == "percent" else 0.0
                  #  ),
                  #  "promo_discount_fixed": (
                  #      r["discount_amount"] if r["discount_type"] == "fixed" else 0.0
                   # ),
                })

    promo_daily_df = pd.DataFrame(expanded_rows)

    if promo_daily_df.empty:
        return pd.DataFrame(
            columns=[
                "ds",
                "Shop_SKU",
                "promo_flag",
             #   "promo_discount_percent",
             #   "promo_discount_fixed",
            ]
        )

    promo_daily_df = (
        promo_daily_df
        .groupby(["ds", "Shop_SKU"], as_index=False)
        .agg(
            promo_flag=("promo_flag", "max"),
          #  promo_discount_percent=("promo_discount_percent", "max"),
           # promo_discount_fixed=("promo_discount_fixed", "max"),
        )
    )

    return promo_daily_df

#7) Build promo to be holiday format df
def build_promo_holidays_df(promo_daily_df, promo_windows: dict):
    """
    promo_daily_df: ds, Shop_SKU, promo_flag, ...
    output: promo_holidays_df with columns holiday, ds, lower_window, upper_window, Shop_SKU
    """
    # if promo_windows is None:
    #     promo_windows = {"lower_window": 0, "upper_window": 0}
    if promo_windows is None:
        raise ValueError("promo_windows must be provided from main(config['promo']).")

    lower = int(promo_windows.get("lower_window", 0))
    upper = int(promo_windows.get("upper_window", 0))

    if promo_daily_df.empty:
        return pd.DataFrame(columns=["holiday","ds","lower_window","upper_window","Shop_SKU"])

    promo_h = promo_daily_df[["ds","Shop_SKU"]].copy()
    promo_h["holiday"] = "promotion"
    promo_h["lower_window"] = promo_windows["lower_window"]
    promo_h["upper_window"] = promo_windows["upper_window"]
    return promo_h

#8) Attach promo Regressor flag to existing forecast sku
def attach_promo_regressors_for_sku(sku_df, promo_daily_df_sku):
    if promo_daily_df_sku is None or promo_daily_df_sku.empty:
        sku_df["promo_flag"] = 0
       # sku_df["promo_discount_percent"] = 0.0
       # sku_df["promo_discount_fixed"] = 0.0
        return sku_df

    cols = ["ds","promo_flag"]
            #,"promo_discount_percent","promo_discount_fixed"]
    promo_tmp = promo_daily_df_sku.reindex(columns=cols)

    sku_df = sku_df.merge(
    promo_tmp,
    on="ds",
    how="left"
    )

    sku_df["promo_flag"] = sku_df["promo_flag"].fillna(0).astype(int)
    #sku_df["promo_discount_percent"] = sku_df["promo_discount_percent"].fillna(0.0)
    #sku_df["promo_discount_fixed"] = sku_df["promo_discount_fixed"].fillna(0.0)
    return sku_df


def forecast_sku_prophet(
    sku,
    join_outbound_df,
    base_holidays_df,
    promo_daily_df_sku,
    promo_holidays_df_sku,
    periods,
    prophet_config: dict | None = None,
    lag_feature_config: dict | None = None,
    return_model_json: bool = False,   # ✅ เพิ่ม
):
    sku_df = join_outbound_df[join_outbound_df["Shop_SKU"] == sku].copy()
    lag_feature_config = lag_feature_config or {}
    lag_days = lag_feature_config.get("lag_days") or [1, 7]
    rolling_mean_windows = lag_feature_config.get("rolling_mean_windows") or [3]
    sku_df, lag_feature_cols = add_lag_features(
        sku_df,
        lag_days=lag_days,
        rolling_mean_windows=rolling_mean_windows,
    )

    # promo regressors (train)
    sku_df = attach_promo_regressors_for_sku(sku_df, promo_daily_df_sku)

    # ✅ DEBUG
    print(
        f"[PROPHET] SKU={sku} | "
        f"promo_rows={0 if promo_daily_df_sku is None else len(promo_daily_df_sku)} | "
        f"promo_days_in_train={int(sku_df['promo_flag'].sum())}"
    )

    # drop rows where lag features are unavailable
    sku_df = sku_df.dropna(subset=lag_feature_cols).copy()

    # holidays for this SKU = base + promo
    holidays_sku = base_holidays_df.copy()
    if promo_holidays_df_sku is not None and not promo_holidays_df_sku.empty:
        holidays_sku = pd.concat(
            [holidays_sku, promo_holidays_df_sku[["holiday","ds","lower_window","upper_window"]]],
            ignore_index=True
        )

    prophet_config = prophet_config or {}
    model = Prophet(
        holidays=holidays_sku,
        weekly_seasonality=bool(prophet_config.get("weekly_seasonality", True)),
        daily_seasonality=bool(prophet_config.get("daily_seasonality", True)),
        seasonality_mode=str(prophet_config.get("seasonality_mode", "multiplicative")),
        changepoint_prior_scale=float(prophet_config.get("changepoint_prior_scale", 0.5)),
        changepoint_range=float(prophet_config.get("changepoint_range", 0.1)),
        seasonality_prior_scale=float(prophet_config.get("seasonality_prior_scale", 5.0)),
    )

    # regressors
    for reg in lag_feature_cols:
        model.add_regressor(reg)

    for reg in ["promo_flag"]:
                #,"promo_discount_percent","promo_discount_fixed"]:
        model.add_regressor(reg)

    model.fit(sku_df[
        ["ds", "y", *lag_feature_cols, "promo_flag"]
    ])
    
    model_json = None
    if return_model_json:
        model_json = model_to_json(model)   # ✅ เพิ่ม

    future = model.make_future_dataframe(periods=periods, freq="D", include_history=True)

    # lag regressors future (ใช้ค่าล่าสุดแบบเดิม)
    last_known = sku_df.iloc[-1]
    for reg in lag_feature_cols:
        future[reg] = last_known[reg]

    # promo regressors future (จาก promo_daily_df_sku)
    if promo_daily_df_sku is None or promo_daily_df_sku.empty:
        future["promo_flag"] = 0
        #future["promo_discount_percent"] = 0.0
        #future["promo_discount_fixed"] = 0.0
    else:
        cols = ["ds","promo_flag"]
                #,"promo_discount_percent","promo_discount_fixed"]
        promo_tmp = promo_daily_df_sku.reindex(columns=cols)

        future = future.merge(
            promo_tmp,
            on="ds",
            how="left"
        )

        future["promo_flag"] = future["promo_flag"].fillna(0).astype(int)
        #future["promo_discount_percent"] = future["promo_discount_percent"].fillna(0.0).astype(float)
        #future["promo_discount_fixed"] = future["promo_discount_fixed"].fillna(0.0).astype(float)

    forecast = model.predict(future)
    forecast["Shop_SKU"] = sku
    out_fcst = forecast[["ds", "Shop_SKU", "yhat", "yhat_lower", "yhat_upper"]].copy()
    
    if return_model_json:
        model_json = model_to_json(model)
        return out_fcst, model_json

    return out_fcst


#9) Logic pick prophet vs Poisson
def pick_prophet_skus(top_skus_80per, promo_daily_df):
    promo_skus = set(promo_daily_df["Shop_SKU"].unique()) if promo_daily_df is not None and not promo_daily_df.empty else set()
    return set(top_skus_80per).union(promo_skus)

#10) Poisson forecasting
def poisson_forecast_with_history(sku, join_outbound_df, periods):
    """
    Generate Poisson forecasts for both historical and future periods, including exact confidence intervals.
    """
    # Filter historical data for the given SKU
    sku_df = join_outbound_df[join_outbound_df['Shop_SKU'] == sku]
    
    if sku_df.empty:
        print(f"Warning: No data found for SKU {sku}. Returning zeros.")
        forecasted_orders = [0] * periods
        future_dates = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq='D')
        return pd.DataFrame({
            'ds': future_dates, 
            'yhat': forecasted_orders, 
            'yhat_lower': [0] * periods, 
            'yhat_upper': [0] * periods, 
            'Shop_SKU': sku
        })
    
    # Generate Poisson predictions for historical periods
    mean_orders_per_day = sku_df['y'].mean()
    if mean_orders_per_day <= 0 or np.isnan(mean_orders_per_day):
        historical_yhat = [0] * len(sku_df)
        historical_lower_bounds = [0] * len(sku_df)
        historical_upper_bounds = [0] * len(sku_df)
    else:
        historical_yhat = poisson.rvs(mu=mean_orders_per_day, size=len(sku_df))
        historical_lower_bounds = [
            max(0, int(poisson.interval(0.95, order)[0])) for order in historical_yhat
        ]
        historical_upper_bounds = [
            int(poisson.interval(0.95, order)[1]) for order in historical_yhat
        ]
    
    historical_forecast = sku_df[['ds', 'Shop_SKU']].copy()
    historical_forecast['yhat'] = historical_yhat
    historical_forecast['yhat_lower'] = historical_lower_bounds
    historical_forecast['yhat_upper'] = historical_upper_bounds
    
    # Generate future Poisson forecasts
    if mean_orders_per_day <= 0 or np.isnan(mean_orders_per_day):
        future_forecasted_orders = [0] * periods
        future_lower_bounds = [0] * periods
        future_upper_bounds = [0] * periods
    else:
        future_forecasted_orders = poisson.rvs(mu=mean_orders_per_day, size=periods)
        future_lower_bounds = [
            max(0, int(poisson.interval(0.95, order)[0])) for order in future_forecasted_orders
        ]
        future_upper_bounds = [
            int(poisson.interval(0.95, order)[1]) for order in future_forecasted_orders
        ]
    
    last_date = sku_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    future_forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_forecasted_orders,
        'yhat_lower': future_lower_bounds,
        'yhat_upper': future_upper_bounds,
        'Shop_SKU': sku
    })
    
    # Combine historical and future forecasts
    return pd.concat([historical_forecast, future_forecast], ignore_index=True)

#11) Forecast all SKu hybrid model parallel
def forecast_all_skus_hybrid_parallel(
    join_outbound_df,
    base_holidays_df,
    promo_daily_df,
    promo_holidays_df,
    prophet_skus,
    periods=14,
    batch_size=50,
    n_jobs=8,
    prophet_config: dict | None = None,
    lag_feature_config: dict | None = None,
    return_models_json: bool = False,   # ✅ เพิ่ม
):
    skus = join_outbound_df["Shop_SKU"].unique()
    all_results = []
    models_json = {}  # ✅ เพิ่ม

    promo_daily_map = {}
    promo_holiday_map = {}
    if promo_daily_df is not None and not promo_daily_df.empty:
        for sku, g in promo_daily_df.groupby("Shop_SKU"):
            promo_daily_map[sku] = g.copy()

    if promo_holidays_df is not None and not promo_holidays_df.empty:
        for sku, g in promo_holidays_df.groupby("Shop_SKU"):
            promo_holiday_map[sku] = g.copy()

    for i in range(0, len(skus), batch_size):
        batch_skus = skus[i:i+batch_size]
        print(f"Processing batch: {i+1}..{min(i+batch_size, len(skus))} / {len(skus)}")

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(forecast_sku_prophet)(
                sku,
                join_outbound_df,
                base_holidays_df,
                promo_daily_map.get(sku, pd.DataFrame()),
                promo_holiday_map.get(sku, pd.DataFrame()),
                periods,
                prophet_config=prophet_config,
                lag_feature_config=lag_feature_config,
                return_model_json=return_models_json,   # ✅ เพิ่ม
            )
            if sku in prophet_skus
            else delayed(poisson_forecast_with_history)(sku, join_outbound_df, periods)
            for sku in batch_skus
        )

        # ✅ unpack
        batch_forecasts = []
        for sku, res in zip(batch_skus, results):
            if (sku in prophet_skus) and return_models_json:
                fcst_df, model_json = res
                batch_forecasts.append(fcst_df)
                if model_json is not None:
                    models_json[sku] = model_json
            else:
                batch_forecasts.append(res)

        #all_results.append(pd.concat(results, ignore_index=True))
        all_results.append(pd.concat(batch_forecasts, ignore_index=True))

    forecast_all = pd.concat(all_results, ignore_index=True)

    # clean int
    for c in ["yhat","yhat_lower","yhat_upper"]:
        if c in forecast_all.columns:
            forecast_all[c] = forecast_all[c].fillna(0).clip(lower=0).round().astype(int)

    if return_models_json:
        return forecast_all, models_json
    return forecast_all

# #12) Filter customer df from prep data fabric
# def filter_customer(oms_pandas_df, customer_code: str, warehousecode: str | None = None):
#     join_outbound_df = oms_pandas_df[oms_pandas_df["customer_code"] == customer_code].copy()

#     if join_outbound_df.empty:
#         raise ValueError(f"No data found for customer_code={customer_code}")
    
#     # ✅ ถ้ามี warehousecode ก็ filter เพิ่ม
#     if warehousecode is not None and "warehousecode" in join_outbound_df.columns:
#         join_outbound_df = join_outbound_df[join_outbound_df["warehousecode"].astype(str) == str(warehousecode)].copy()

#     # ✅ keep warehousecode ไว้ (ถ้ามี)
#     cols = ["ds", "Shop_SKU", "y"]
#     if "warehousecode" in join_outbound_df.columns:
#         cols.append("warehousecode")
#     return join_outbound_df[cols]

def filter_customer(
    oms_pandas_df: pd.DataFrame,
    customer_code: str,
    warehousecode: str | None = None,
) -> pd.DataFrame:
    """
    Return df with columns: ds, Shop_SKU, y (+ warehousecode if exists)

    Behavior:
    - If no rows for customer_code at all -> raise ValueError (hard error)
    - If warehousecode is provided but no rows match -> return EMPTY df (soft skip)
    """

    # 1) filter customer
    join_outbound_df = oms_pandas_df[oms_pandas_df["customer_code"].astype(str) == str(customer_code)].copy()
    if join_outbound_df.empty:
        raise ValueError(f"No data found for customer_code={customer_code}")

    # 2) optional warehouse filter
    if warehousecode is not None:
        if "warehousecode" not in join_outbound_df.columns:
            # ไม่มีคอลัมน์ warehousecode ใน OMS ตอนนี้
            # → ใช้ข้อมูลทุก order ของลูกค้านี้ แล้วไป stamp warehousecode ที่ท้าย flow แทน
            print(
                f"[INFO] column 'warehousecode' not found in OMS for customer_code={customer_code} "
                f"-> use all rows for this customer and stamp warehousecode={warehousecode} later"
            )
        else:
            join_outbound_df = join_outbound_df[
                join_outbound_df["warehousecode"].astype(str) == str(warehousecode)
            ].copy()

            if join_outbound_df.empty:
                # ✅ สำคัญ: กันคลังไม่มีข้อมูล -> ไม่ raise ให้ลูปหยุด, คืน df ว่างแทน
                print(f"[SKIP] No data for customer_code={customer_code} warehousecode={warehousecode} -> return empty")
                empty_cols = ["ds", "Shop_SKU", "y", "warehousecode"]
                return pd.DataFrame(columns=empty_cols)

    # 3) select columns (keep warehousecode if exists)
    cols = ["ds", "Shop_SKU", "y"]
    if "warehousecode" in join_outbound_df.columns:
        cols.append("warehousecode")

    return join_outbound_df[cols].reset_index(drop=True)


#14) Build promo inspection output
def build_promo_inspection_outputs(forecast_df_all: pd.DataFrame, promo_daily_df: pd.DataFrame):
    """
    Returns:
      promo_skus: list[str]
      promo_start, promo_end: Timestamp
      promo_inspect_forecast: DataFrame
      promo_vs_nonpromo_summary: DataFrame
    """
    if promo_daily_df is None or promo_daily_df.empty or "promo_flag" not in promo_daily_df.columns:
        # no promo
        empty = pd.DataFrame()
        return [], None, None, empty, empty

    promo_daily_df = promo_daily_df.copy()
    promo_daily_df["ds"] = pd.to_datetime(promo_daily_df["ds"], errors="coerce").dt.normalize()
    promo_daily_df["Shop_SKU"] = promo_daily_df["Shop_SKU"].astype(str)

    promo_active = promo_daily_df[promo_daily_df["promo_flag"] == 1].copy()
    if promo_active.empty:
        empty = pd.DataFrame()
        return [], None, None, empty, empty

    promo_skus = promo_active["Shop_SKU"].unique().tolist()
    promo_start = promo_active["ds"].min()
    promo_end   = promo_active["ds"].max()

    f = forecast_df_all.copy()
    f["ds"] = pd.to_datetime(f["ds"], errors="coerce").dt.normalize()
    f["Shop_SKU"] = f["Shop_SKU"].astype(str)

    # --- 1) promo window forecast rows (for promo SKUs only)
    promo_inspect_forecast = (
        f.loc[
            f["Shop_SKU"].isin(promo_skus) &
            f["ds"].between(promo_start, promo_end)
        ]
        .sort_values(["Shop_SKU", "ds"])
        .reset_index(drop=True)
    )

    # --- 2) compare promo vs non-promo (summary)
    # Use promo_flag already merged into forecast_df_all (if not present, assume 0)
    if "promo_flag" not in promo_inspect_forecast.columns:
        # If you didn't merge promo_flag into forecast_df_all, we can still join here,
        # but your current main already merges promo_flag, so normally not needed.
        pass

    promo_vs_nonpromo_summary = (
        f.loc[f["Shop_SKU"].isin(promo_skus)]
        .assign(is_promo=lambda x: x.get("promo_flag", 0).fillna(0).astype(int) == 1)
        .groupby(["Shop_SKU", "is_promo"])
        .agg(
            avg_yhat=("yhat", "mean"),
            median_yhat=("yhat", "median"),
            n_days=("yhat", "count")
        )
        .reset_index()
        .sort_values(["Shop_SKU", "is_promo"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return promo_skus, promo_start, promo_end, promo_inspect_forecast, promo_vs_nonpromo_summary

#15) Supabase S3 for Upload JSON
# ใช้ shared S3 client ต่อ process (lazy-init)
_S3_CLIENT: Optional[Any] = None


def _get_supabase_s3_client():
    """
    Lazily create and cache a boto3 S3 client for Supabase Storage.
    """
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT

    supabase_url = os.getenv("SUPABASE_URL")
    region = os.getenv("SUPABASE_REGION")
    aws_key_id = os.getenv("SUPABASE_AWS_KEY_ID")
    aws_secret = os.getenv("SUPABASE_AWS_SECRET_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is not set")
    if not aws_key_id or not aws_secret:
        raise ValueError("SUPABASE_AWS_KEY_ID / SUPABASE_AWS_SECRET_KEY not set")

    # ✅ build endpoint_url correctly (avoid double /storage/v1/s3)
    if supabase_url.endswith("/storage/v1/s3"):
        endpoint_url = supabase_url
    else:
        endpoint_url = f"{supabase_url}/storage/v1/s3"

    _S3_CLIENT = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_key_id,
        aws_secret_access_key=aws_secret,
        config=Config(signature_version="s3v4"),
    )

    # debug (ควรเห็นแค่ /storage/v1/s3 รอบเดียว)
    print("S3 client endpoint =", _S3_CLIENT.meta.endpoint_url)
    return _S3_CLIENT


def supabase_s3_upload_json(bucket: str, object_path: str, json_str: str):
    """
    Upload JSON string to Supabase S3 using a shared S3 client.
    """
    s3 = _get_supabase_s3_client()

    body = json_str.encode("utf-8")
    print(f"[UPLOAD] bucket={bucket} key={object_path} bytes={len(body)}")
    try:
        s3.put_object(
            Bucket=bucket,
            Key=object_path,
            Body=body,
            ContentType="application/json",
            ContentLength=len(body),
        )
    except ClientError as e:
        print("Upload failed:", e.response.get("Error", {}))
        raise

# ---------------------------------------------------------------------------
# 3) Training orchestration
# ---------------------------------------------------------------------------
# Main function for end-to-end forecast flow
def main(
    *,
    customer_code: str,
    warehousecode: str | None = None,   # ✅ ADD
    oms_pandas_df: pd.DataFrame,
    holiday_df: pd.DataFrame | None = None,
    promo_json_list: list | None = None,
    config: dict | None = None,
    periods_base: int = 14,
    test_days: int | None = None,
    batch_size: int | None = None,
    n_jobs: int | None = None,
    return_models_json: bool = False,
    prophet_skus_override: list[str] | None = None,   # ✅ ADD
):
    print("[MAIN] start")

    # ---------------------------
    # 0) safety checks
    # ---------------------------
    total, used, free = shutil.disk_usage("/")
    print(f"[MAIN] Disk Space - Total: {total // (2**30)} GB, Used: {used // (2**30)} GB, Free: {free // (2**30)} GB")
    if free < 1:
        raise RuntimeError("Insufficient disk space to run the script.")
    print("[MAIN] after disk check")

    # ---------------------------
    # 0.1) config normalize (single source of truth in config.py)
    # ---------------------------
    runtime_config = normalize_training_config(config)
    execution_cfg = runtime_config.get("execution", {})
    impact_cfg = runtime_config.get("impact_dates", {})
    lag_feature_cfg = runtime_config.get("lag_features", {})
    promo_cfg = runtime_config.get("promo", {})
    prophet_cfg = runtime_config.get("advanced", {}).get("prophet", {})

    test_days = int(test_days) if test_days is not None else int(execution_cfg.get("test_days", 30))
    batch_size = int(batch_size) if batch_size is not None else int(execution_cfg.get("batch_size", 50))
    n_jobs = int(n_jobs) if n_jobs is not None else int(execution_cfg.get("n_jobs", 8))

    print("[MAIN] effective runtime_config:", runtime_config)

    # ---------------------------
    # 1) normalize input dates
    # ---------------------------
    # join_outbound_df = preprocess_all(outbound_df)
    join_outbound_df = filter_customer(oms_pandas_df, customer_code=customer_code, warehousecode=warehousecode)
    join_outbound_df = join_outbound_df.copy()

    join_outbound_df["ds"] = pd.to_datetime(join_outbound_df["ds"], errors="coerce").dt.normalize()

    # basic sanity
    if join_outbound_df["ds"].isna().all():
        raise ValueError("join_outbound_df['ds'] parse failed (all NaT). Check date format.")
    if "Shop_SKU" not in join_outbound_df.columns or "y" not in join_outbound_df.columns:
        raise ValueError("join_outbound_df must have columns: ['ds','Shop_SKU','y']")

    print("[MAIN] join_outbound_df:", join_outbound_df.shape, 
          "date_range:", join_outbound_df["ds"].min(), "->", join_outbound_df["ds"].max())

    if holiday_df is None:
        raise ValueError("holiday_df is required (got None). Please pass get_holiday_df() result.")

    holiday_df_local = holiday_df.copy()

    # Supabase ds มักเป็น object/string
    holiday_df_local["ds"] = pd.to_datetime(holiday_df_local["ds"], errors="coerce").dt.normalize()
    if holiday_df_local["ds"].isna().any():
        n_bad = int(holiday_df_local["ds"].isna().sum())
        print(f"[MAIN] ⚠️ holiday_df has {n_bad} rows with invalid ds -> dropped")

    # Ensure columns exist
    if "lower_window" not in holiday_df_local.columns:
        holiday_df_local["lower_window"] = 0
    if "upper_window" not in holiday_df_local.columns:
        holiday_df_local["upper_window"] = 0

    # Prophet requires: holiday, ds, lower_window, upper_window
    base_holidays_df = (
        holiday_df_local
        .rename(columns={"holiday_code": "holiday"})
        [["holiday", "ds", "lower_window", "upper_window"]]
        .dropna(subset=["holiday", "ds"])
        .drop_duplicates(subset=["holiday", "ds", "lower_window", "upper_window"])
        .sort_values(["holiday", "ds"])
        .reset_index(drop=True)
    )
    print("[MAIN] base_holidays_df (from Supabase):", base_holidays_df.shape)

    # Filter holidays by per-customer boolean flags from config
    _holiday_cfg = runtime_config.get("holiday", {})
    _holiday_flag_map = {
        "doubleday_peak": _holiday_cfg.get("use_doubleday_peak", True),
        "holiday":        _holiday_cfg.get("use_public_holiday", True),
        "midmonth_dates": _holiday_cfg.get("use_midmonth_dates", True),
        "payday_dates":   _holiday_cfg.get("use_payday_dates",   True),
    }
    _enabled_codes = {code for code, enabled in _holiday_flag_map.items() if enabled}
    base_holidays_df = base_holidays_df[base_holidays_df["holiday"].isin(_enabled_codes)].copy()
    print(f"[MAIN] holidays after flag filter: {sorted(base_holidays_df['holiday'].unique())}")

    # ---------------------------
    # 2) train/test split
    # ---------------------------
    latest_date = join_outbound_df["ds"].max()
    train_df = join_outbound_df[join_outbound_df["ds"] <= latest_date - pd.Timedelta(days=1)]
    test_df  = join_outbound_df[join_outbound_df["ds"] >  latest_date - pd.Timedelta(days=test_days)]

    print("[MAIN] train_df:", train_df.shape, "test_df:", test_df.shape, "latest_date:", latest_date)

    if train_df.empty:
        raise ValueError("train_df is empty. Check your ds range and data availability.")

    # ---------------------------
    # 3) top80% SKUs by volume (from train)
    # ---------------------------
    sku_volume = train_df.groupby("Shop_SKU")["y"].sum().sort_values(ascending=False)
    if sku_volume.sum() <= 0:
        raise ValueError("Total volume is 0. Check y values.")

    cumulative = sku_volume.cumsum() / sku_volume.sum()
    top_skus_80per = cumulative[cumulative <= 0.8].index.tolist()
    if len(top_skus_80per) == 0:
        # fallback: at least take top 1
        top_skus_80per = [sku_volume.index[0]]

    print("[MAIN] n_all_skus:", join_outbound_df["Shop_SKU"].nunique(), 
          "n_top80:", len(top_skus_80per))

    # ---------------------------
    # 4) impact_dates -> impact_holidays -> concat into base_holidays_df
    # ---------------------------
    print("[MAIN] Extracting impact dates (CoV + Z rolling)...")
    impact_dates = extract_impact_dates_with_cv(
        train_df,
        top_skus_n=int(impact_cfg.get("top_skus_n", 30)),
        cv_bands=impact_cfg.get("cv_bands"),
        zscore_rules=impact_cfg.get("zscore_rules"),
    )

    # build impact holidays with config window
    impact_holidays_df = build_impact_holidays_df(
        impact_dates, 
        {"impact_dates": impact_cfg}
    )

    base_holidays_df = (
        pd.concat([base_holidays_df, impact_holidays_df], ignore_index=True)
        .drop_duplicates(subset=["holiday", "ds", "lower_window", "upper_window"])
        .sort_values(["holiday", "ds"])
        .reset_index(drop=True)
    )
    print("[MAIN] base_holidays_df (+impact):", base_holidays_df.shape)

    # ---------------------------
    # 5) promotions (optional)
    # ---------------------------
    all_skus = join_outbound_df["Shop_SKU"].unique().tolist()
    promo_json_list = promo_json_list or []
    use_promotions = bool(promo_cfg.get("use_promotions", True))

    if not use_promotions:
        print("[MAIN] promotions disabled by config (promo.use_promotions=false)")
        promo_json_list = []

    if len(promo_json_list) > 0:
        promo_master_df = normalize_promotions(promo_json_list)
        promo_daily_df = build_promo_daily_df(promo_master_df, all_skus)
    else:
        promo_master_df = pd.DataFrame()
        promo_daily_df = pd.DataFrame(columns=["ds","Shop_SKU","promo_flag"])
                                               #,"promo_discount_percent","promo_discount_fixed"])

    promo_holidays_df = build_promo_holidays_df(
        promo_daily_df, 
        promo_windows=promo_cfg
    )

    print("[MAIN] promo_master_df:", promo_master_df.shape,
          "promo_daily_df:", promo_daily_df.shape)

    # ---------------------------
    # 6) prophet sku selection = top80 ∪ promo_skus
    # ---------------------------
    # 6) prophet sku selection
    # ---------------------------
    if prophet_skus_override is not None and len(prophet_skus_override) > 0:
        # unique preserve order
        seen = set()
        prophet_skus = []
        for s in prophet_skus_override:
            if s not in seen:
                prophet_skus.append(s)
                seen.add(s)
        print("[MAIN] prophet_skus OVERRIDDEN:", len(prophet_skus), "example:", prophet_skus[:5])
    else:
        prophet_skus = pick_prophet_skus(top_skus_80per, promo_daily_df)
        print("[MAIN] n_prophet_skus:", len(prophet_skus), "example:", list(prophet_skus)[:5])

    #### FILTER ONLY PROMO SKU to TRAIN
    # ✅ OPTIONAL: forecast only these SKUs (not all 1431)
    if prophet_skus_override is not None and len(prophet_skus_override) > 0:
        join_outbound_df = join_outbound_df[join_outbound_df["Shop_SKU"].isin(prophet_skus)].copy()
        train_df = train_df[train_df["Shop_SKU"].isin(prophet_skus)].copy()
        test_df  = test_df[test_df["Shop_SKU"].isin(prophet_skus)].copy()

        # promo_daily_df ก็ filter ให้เล็กลงด้วย (กัน merge ใหญ่)
        promo_daily_df = promo_daily_df[promo_daily_df["Shop_SKU"].isin(prophet_skus)].copy()

        print("[MAIN] ✅ FILTERED to prophet_skus only. join_outbound_df:", join_outbound_df.shape)



    # ---------------------------
    # 7) periods_effective (missing days)
    # ---------------------------
    today = pd.Timestamp.today().normalize()
    dif_days, last_ds, expected_last = compute_missing_days(join_outbound_df, today=today)
    periods_effective = periods_base + dif_days

    print(f"[MAIN] last_ds={last_ds} expected_last={expected_last} dif_days={dif_days} periods_effective={periods_effective}")
    print(f"[MAIN] forecast settings: n_jobs={n_jobs} batch_size={batch_size}")

    # ---------------------------
    # 8) forecast hybrid (Prophet or Poisson)
    # ---------------------------
    print("[MAIN] Starting forecast_all_skus_hybrid_parallel ...")
    res = forecast_all_skus_hybrid_parallel(
    join_outbound_df=join_outbound_df,
    base_holidays_df=base_holidays_df,
    promo_daily_df=promo_daily_df,
    promo_holidays_df=promo_holidays_df,
    prophet_skus=prophet_skus,
    periods=periods_effective,
    batch_size=batch_size,
    n_jobs=n_jobs,
    prophet_config=prophet_cfg,
    lag_feature_config=lag_feature_cfg,
    return_models_json=return_models_json,
)
    if return_models_json:
        forecast_df_all, prophet_models_json = res
    else:
        forecast_df_all = res
        prophet_models_json = None


    print("[MAIN] forecast_df_all:", forecast_df_all.shape)
    forecast_df_all = forecast_df_all.sort_values(["ds","Shop_SKU"]).reset_index(drop=True)
    # ---------------------------
    # 9) split past/future
    # ---------------------------
    forecast_df_all["ds"] = pd.to_datetime(forecast_df_all["ds"], errors="coerce").dt.normalize()
    forecast_df_past = forecast_df_all[forecast_df_all["ds"] <= latest_date].copy()
    forecast_df_future = forecast_df_all[forecast_df_all["ds"] > latest_date].copy()
    
    # future: today -> today+13
    start_prod = today
    end_prod = today + pd.Timedelta(days=13)
    forecast_df_future = forecast_df_future[
        (forecast_df_future["ds"] >= start_prod) & (forecast_df_future["ds"] <= end_prod)
    ].copy()

    # test_df: exclude missing days
    test_df = test_df[pd.to_datetime(test_df["ds"], errors="coerce").dt.normalize() <= last_ds]

    # ---------------------------
    # 10) bias correction (ใช้ past)
    # ---------------------------
    print("[MAIN] Applying correction on test period...")
    correction_map = compute_bias_correction(
        test_df, forecast_df_past, min_demand_threshold=5000, min_positive_days=180
    )

    forecast_df_past = apply_correction(forecast_df_past, correction_map)
    forecast_df_past["yhat"] = forecast_df_past["yhat_corrected"].fillna(forecast_df_past["yhat"])
    forecast_df_past.drop(columns=["yhat_corrected"], inplace=True, errors="ignore")

    forecast_df_future = apply_correction(forecast_df_future, correction_map)
    forecast_df_future["yhat"] = forecast_df_future["yhat_corrected"].fillna(forecast_df_future["yhat"])
    forecast_df_future.drop(columns=["yhat_corrected"], inplace=True, errors="ignore")

    # รวมกลับ
    forecast_df_all = pd.concat([forecast_df_past, forecast_df_future], ignore_index=True)
    forecast_df_all = forecast_df_all.sort_values(["ds","Shop_SKU"]).reset_index(drop=True)
    # ==============================
    # ✅ MERGE PROMO INFO HERE
    # ==============================
    promo_cols = [
        "ds", "Shop_SKU",
        "promo_flag",
    #    "promo_discount_percent",
    #    "promo_discount_fixed"
    ]

    forecast_df_all = forecast_df_all.merge(
        promo_daily_df[promo_cols],
        on=["ds", "Shop_SKU"],
        how="left"
    )

    forecast_df_all["promo_flag"] = forecast_df_all["promo_flag"].fillna(0).astype(int)

    forecast_df_all = forecast_df_all.sort_values(["ds","Shop_SKU"]).reset_index(drop=True)
    #forecast_df_all["promo_discount_percent"] = forecast_df_all["promo_discount_percent"].fillna(0.0)
    #forecast_df_all["promo_discount_fixed"] = forecast_df_all["promo_discount_fixed"].fillna(0.0)
    # ==============================
    # ✅ DYNAMIC PROMO INSPECTION (NO HARDCODE)
    # ==============================
    promo_skus, promo_start, promo_end, promo_inspect_forecast, promo_vs_nonpromo_summary = \
        build_promo_inspection_outputs(forecast_df_all, promo_daily_df)

    if promo_skus:
        print(f"[MAIN] Promo SKUs for inspection: {promo_skus}")
        print(f"[MAIN] Promo date range: {promo_start} -> {promo_end}")
        # ถ้าอยากดูตัวอย่าง
        #display(promo_inspect_forecast[["ds", "Shop_SKU", "yhat", "promo_flag"]].head(50))
        #display(promo_vs_nonpromo_summary.head(50))
    else:
        print("[MAIN] No active promo rows found for inspection.")

    # ==============================
    # THEN SPLIT FUTURE / PAST
    # ==============================
    forecast_df_future = forecast_df_all[forecast_df_all["ds"] >latest_date].copy()
    forecast_df_past   = forecast_df_all[forecast_df_all["ds"] <=  latest_date].copy()



    # ---------------------------
    # 11) post-processing int
    # ---------------------------
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        if c in forecast_df_all.columns:
            forecast_df_all[c] = (
                forecast_df_all[c].fillna(0)
                .replace([np.inf, -np.inf], 0)
                .clip(lower=0)
                .round()
                .astype(int)
            )

    print("[MAIN] done")
    return (
        train_df,
        test_df,
        top_skus_80per,
        impact_dates,
        base_holidays_df,
        promo_daily_df,
        prophet_skus,
        forecast_df_all,
        forecast_df_future,
        forecast_df_past,
        promo_inspect_forecast,         # ✅ NEW
        promo_vs_nonpromo_summary  ,     # ✅ NEW
        prophet_models_json if return_models_json else None,
    )

# Call main + train + upload Prophet model JSON
def safe_key(x: str) -> str:
    return "".join(c if c.isalnum() or c in ["_", "-", "."] else "_" for c in str(x))

def train_forecast_and_upload_models(
    customer_code: str,
    oms_pandas_df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    promo_json_list: list | None = None,
    config: dict | None = None,
    periods: int = 14,          # ✅ ADD
    promo_only: bool = True,    # ✅ (ถ้าอยากใช้)
    warehousecode: str | None = None,   # ✅ ADD
):
    promo_skus = []
    if promo_only:
        if not promo_json_list:
            raise ValueError("promo_only=True but promo_json_list is None/empty")
        promo_skus = extract_skus_from_promo_json(promo_json_list)
        if not promo_skus:
            raise ValueError("No SKU found in promo_json_list['include_sku'].")
    print("[WRAPPER] promo_only:", promo_only, "promo_skus:", promo_skus)

    out = main(
        customer_code=customer_code,
        warehousecode=warehousecode,
        oms_pandas_df=oms_pandas_df,
        holiday_df=holiday_df,
        promo_json_list=promo_json_list,
        config=config,
         periods_base=periods,                 # ✅ ใช้ periods_base เดิมของ main
        return_models_json=True,
        prophet_skus_override=promo_skus if promo_only else None,  # ✅ ต้องให้ main รับได้
    )

    forecast_df_all = out[7]
    forecast_df_future = out[8]
    prophet_models_json = out[12]

    # ✅ stamp customer_code / warehousecode ให้ติดออกมาตั้งแต่ต้นทาง
    for _df in (forecast_df_all, forecast_df_future):
        if _df is None or _df.empty:
            continue

        # customer_code
        if "customer_code" not in _df.columns and "customercode" not in _df.columns:
            _df["customer_code"] = str(customer_code)
        elif "customer_code" in _df.columns:
            _df["customer_code"] = _df["customer_code"].astype(str)
        elif "customercode" in _df.columns:
            _df["customercode"] = _df["customercode"].astype(str)

        # warehousecode (ถ้ารันแบบระบุคลัง)
        if warehousecode:
            if "warehousecode" not in _df.columns and "warehouse_code" not in _df.columns:
                _df["warehousecode"] = str(warehousecode)
            elif "warehousecode" in _df.columns:
                _df["warehousecode"] = _df["warehousecode"].astype(str)
            elif "warehouse_code" in _df.columns:
                _df["warehouse_code"] = _df["warehouse_code"].astype(str)

    bucket = os.getenv("SUPABASE_MODEL_BUCKET")
    if not bucket:
        raise ValueError("SUPABASE_MODEL_BUCKET is not set")

    if not isinstance(prophet_models_json, dict):
        raise TypeError(f"prophet_models_json must be dict, got {type(prophet_models_json)}")

    uploaded = 0
    for sku, json_str in prophet_models_json.items():
        if warehousecode:
            object_path = f"{safe_key(customer_code)}/{safe_key(warehousecode)}/{safe_key(sku)}.json"
        else:
            object_path = f"{safe_key(customer_code)}/prophet/{safe_key(sku)}.json"

        supabase_s3_upload_json(bucket=bucket, object_path=object_path, json_str=json_str)
        uploaded += 1

    print(f"[DONE] Uploaded {uploaded} Prophet models for customer={customer_code}")
    return forecast_df_all, forecast_df_future

#18) Extract SKU from promotion list json
def extract_skus_from_promo_json(promo_json_list: list) -> list[str]:
    skus = set()
    for p in promo_json_list or []:
        for s in p.get("include_sku", []) or []:
            code = s.get("sku_code")
            if code:
                skus.add(str(code))
    return sorted(skus)


# Promo-only train flow (run + upload model)
def predict_from_promo_json_list(
    *,
    customer_code: str,
    promo_json_list: list,
    periods: int = 14,
):
    """
    Promo-only retrain:
    - load oms + holiday inside
    - config=None (main will normalize default)
    - promo_only=True
    """
    oms_pandas_df = get_oms_pandas_df()
    holiday_df = get_holiday_df()

    forecast_df_all, forecast_df_future = train_forecast_and_upload_models(
        customer_code=customer_code,
        oms_pandas_df=oms_pandas_df,
        holiday_df=holiday_df,
        promo_json_list=promo_json_list,
        config=None,          # ✅ main handles default
        periods=periods,
        promo_only=True,
    )

    return {
        "customer_code": customer_code,
        "periods": periods,
        "promo_skus": extract_skus_from_promo_json(promo_json_list),
        "status": "uploaded",
    }


def compute_missing_days(join_outbound_df, today=None):
    """
    คำนวณจำนวนวันข้อมูลหายไป (dif_days)
    สมมติระบบควรมีข้อมูลถึงเมื่อวาน (today-1)
    """
    if today is None:
        today = pd.Timestamp.today().normalize()  # ตัดเวลาเหลือเป็นวัน
    else:
        today = pd.to_datetime(today).normalize()

    last_ds = pd.to_datetime(join_outbound_df["ds"]).max()
    if pd.isna(last_ds):
        return 0, None, today - pd.Timedelta(days=1)

    last_ds = pd.to_datetime(last_ds).normalize()
    expected_last = today - pd.Timedelta(days=1)

    dif_days = (expected_last - last_ds).days
    dif_days = max(0, dif_days)

    return dif_days, last_ds, expected_last

def run_full_training(
    *,
    customer_code: str,
    periods: int = 14,
    warehousecodes: list[str] | None = None,   # ✅ ADD
):
    oms_pandas_df = get_oms_pandas_df()
    holiday_df = get_holiday_df()

    # ✅ auto list warehouses ถ้าไม่ส่งมา
    if warehousecodes is None:
        if "warehousecode" not in oms_pandas_df.columns:
            warehousecodes = [None]  # ไม่มีคอลัมน์ ก็รันแบบเดิม
        else:
            warehousecodes = (
                oms_pandas_df.loc[oms_pandas_df["customer_code"] == customer_code, "warehousecode"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

    all_future = []
    all_all = []

    for wh in warehousecodes:
        print(f"\n[RUN] customer={customer_code} warehouse={wh}")

        try:
            forecast_df_all, forecast_df_future = train_forecast_and_upload_models(
                customer_code=customer_code,
                oms_pandas_df=oms_pandas_df,
                holiday_df=holiday_df,
                promo_json_list=None,
                config=None,
                periods=periods,
                promo_only=False,
                warehousecode=wh,   # ✅ IMPORTANT (อันอื่นคุณแก้แล้ว)
            )

            # ✅ กันกรณีคลังนี้ไม่มีข้อมูลแล้วฟังก์ชันคืน df ว่าง
            if forecast_df_all is None or forecast_df_all.empty:
                print(f"[SKIP] no output for warehouse={wh}")
                continue

            all_all.append(forecast_df_all)
            if forecast_df_future is not None and not forecast_df_future.empty:
                all_future.append(forecast_df_future)

        except Exception as e:
            # ✅ กันไม่ให้คลังเดียวทำให้ทั้งรอบล้ม (อยากให้ strict ก็ลบบรรทัดนี้ได้)
            print(f"[SKIP] warehouse={wh} failed: {e}")
            continue

    # ✅ กัน concat พังถ้าทุกคลังถูก skip
    out_all = pd.concat(all_all, ignore_index=True) if all_all else pd.DataFrame()
    out_future = pd.concat(all_future, ignore_index=True) if all_future else pd.DataFrame()
    return out_all, out_future


# ---------------------------------------------------------------------------
# 4) Output shaping + upload
# ---------------------------------------------------------------------------
def to_replenishment_forecast_upload_df(
    forecast_df_future: pd.DataFrame,
) -> pd.DataFrame:

    if forecast_df_future is None or forecast_df_future.empty:
        return pd.DataFrame(
            columns=["customercode", "warehousecode", "Shop_SKU", "ds", "yhat"]
        )

    df = forecast_df_future.copy()

    # rename ให้ตรง table
    df = df.rename(columns={
        "customer_code": "customercode",
        "warehouse_code": "warehousecode",
        "shop_sku": "Shop_SKU",
        "ShopSKU": "Shop_SKU",
    })

    # ✅ กันเคสยังไม่มี customercode (บาง flow ไม่ stamp มา)
    if "customercode" not in df.columns:
        raise KeyError("forecast_df_future ไม่มีคอลัมน์ customer_code/customercode")

    # ✅ warehousecode อาจไม่มี ถ้ารันแบบเดิม
    if "warehousecode" not in df.columns:
        df["warehousecode"] = None

    if "Shop_SKU" not in df.columns:
        raise KeyError("forecast_df_future ไม่มีคอลัมน์ Shop_SKU")

    if "ds" not in df.columns:
        raise KeyError("forecast_df_future ไม่มีคอลัมน์ ds")

    if "yhat" not in df.columns:
        raise KeyError("forecast_df_future ไม่มีคอลัมน์ yhat")

    # cast type กันพลาด
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", utc=True)
    df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce").fillna(0).astype(int)

    # เลือกเฉพาะ column ที่ต้อง insert
    out = df[
        ["ds","customercode", "warehousecode", "Shop_SKU", "yhat"]
    ].dropna(subset=["ds"])

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5) Runtime config + scope resolution (อ่าน config/scope จาก DB)
# ---------------------------------------------------------------------------
# NOTE: Use shared Supabase engine from demand_forecast.database_engine.db_manager


def _deep_merge_runtime(base: dict | None, override: dict | None) -> dict:
    merged = dict(base or {})
    if not override:
        return merged
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_runtime(merged.get(key), value)
        else:
            merged[key] = value
    return merged


def _to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_list_or_none(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    out: list[int] = []
    for item in value:
        parsed = _to_int_or_none(item)
        if parsed is None:
            continue
        if parsed > 0 and parsed not in out:
            out.append(parsed)
    return out or None


def _build_config_from_db_row(row: Mapping[str, Any]) -> dict:
    cfg: dict[str, Any] = {}

    # holidays
    holiday_cfg: dict[str, Any] = {}
    for flag in ("use_doubleday_peak", "use_public_holiday", "use_midmonth_dates", "use_payday_dates"):
        val = row.get(flag)
        if val is not None:
            holiday_cfg[flag] = bool(val)
    if holiday_cfg:
        cfg["holiday"] = holiday_cfg

    # promo toggle
    promo_cfg: dict[str, Any] = {}
    if row.get("use_promotions") is not None:
        promo_cfg["use_promotions"] = bool(row.get("use_promotions"))
    if promo_cfg:
        cfg["promo"] = promo_cfg

    # lag features
    lag_cfg: dict[str, Any] = {}
    lag_days = _to_int_list_or_none(row.get("lag_days"))
    rolling_mean_windows = _to_int_list_or_none(row.get("rolling_mean_windows"))
    if lag_days is not None:
        lag_cfg["lag_days"] = lag_days
    if rolling_mean_windows is not None:
        lag_cfg["rolling_mean_windows"] = rolling_mean_windows
    if lag_cfg:
        cfg["lag_features"] = lag_cfg

    # advanced / impact_dates
    advanced_cfg: dict[str, Any] = {}
    impact_cfg: dict[str, Any] = {}
    top_skus_n = _to_int_or_none(row.get("impact_top_skus_n"))
    impact_lw = _to_int_or_none(row.get("impact_lower_window"))
    impact_uw = _to_int_or_none(row.get("impact_upper_window"))
    impact_cv_band_low = _to_float_or_none(row.get("impact_cv_band_low"))
    impact_cv_band_high = _to_float_or_none(row.get("impact_cv_band_high"))
    if top_skus_n is not None:
        impact_cfg["top_skus_n"] = top_skus_n
    if impact_lw is not None:
        impact_cfg["holiday_lower_window"] = impact_lw
    if impact_uw is not None:
        impact_cfg["holiday_upper_window"] = impact_uw
    if impact_cv_band_low is not None and impact_cv_band_high is not None:
        impact_cfg["cv_bands"] = [impact_cv_band_low, impact_cv_band_high]
    if impact_cfg:
        advanced_cfg["impact_dates"] = impact_cfg

    # advanced / prophet
    prophet_cfg: dict[str, Any] = {}
    if row.get("prophet_weekly_seasonality") is not None:
        prophet_cfg["weekly_seasonality"] = bool(row.get("prophet_weekly_seasonality"))
    if row.get("prophet_daily_seasonality") is not None:
        prophet_cfg["daily_seasonality"] = bool(row.get("prophet_daily_seasonality"))
    if row.get("prophet_seasonality_mode") is not None:
        prophet_cfg["seasonality_mode"] = str(row.get("prophet_seasonality_mode"))
    cps = _to_float_or_none(row.get("prophet_changepoint_prior_scale"))
    cpr = _to_float_or_none(row.get("prophet_changepoint_range"))
    sps = _to_float_or_none(row.get("prophet_seasonality_prior_scale"))
    if cps is not None:
        prophet_cfg["changepoint_prior_scale"] = cps
    if cpr is not None:
        prophet_cfg["changepoint_range"] = cpr
    if sps is not None:
        prophet_cfg["seasonality_prior_scale"] = sps
    if prophet_cfg:
        advanced_cfg["prophet"] = prophet_cfg

    if advanced_cfg:
        cfg["advanced"] = advanced_cfg

    return cfg


def load_demand_forecast_config_for(
    customercode: str,
    warehousecode: str | None = None,
    engine=None,
) -> dict:
    """
    Load demand-forecast config from Supabase table warehouse.config_demand_forecast.
    Priority:
      1) exact (customercode, warehousecode)
      2) customer default (customercode, warehousecode IS NULL)
    Return {} if nothing found.
    """
    if engine is None:
        engine = db_manager.supabase_engine

    sql = text(
        """
        SELECT
            customercode,
            warehousecode,
            lag_days,
            rolling_mean_windows,
            impact_top_skus_n,
            impact_lower_window,
            impact_upper_window,
            impact_cv_band_low,
            impact_cv_band_high,
            prophet_weekly_seasonality,
            prophet_daily_seasonality,
            prophet_seasonality_mode,
            prophet_changepoint_prior_scale,
            prophet_changepoint_range,
            prophet_seasonality_prior_scale,
            use_promotions,
            use_doubleday_peak,
            use_public_holiday,
            use_midmonth_dates,
            use_payday_dates
        FROM warehouse.config_demand_forecast
        WHERE is_active = true
          AND (
                (customercode = :cust AND warehousecode = :wh)
             OR (customercode = :cust AND warehousecode IS NULL)
          )
        ORDER BY
          CASE WHEN warehousecode IS NULL THEN 2 ELSE 1 END,
          updated_at DESC
        LIMIT 1
        """
    )

    with engine.connect() as conn:
        row = conn.execute(sql, {"cust": str(customercode), "wh": warehousecode}).mappings().first()

    if not row:
        return {}
    return _build_config_from_db_row(row)


def load_training_scope_df(engine=None) -> pd.DataFrame:
    """
    Load training scope pairs from warehouse.replenishment_outbound_gi.

    Expected output columns:
    - customercode
    - customername
    - warehousecode
    - warehousename
    """
    if engine is None:
        engine = db_manager.supabase_engine

    scope_sql = text(
        """
        SELECT DISTINCT
            customercode::text AS customercode,
            customername::text AS customername,
            warehousecode::text AS warehousecode,
            warehousename::text AS warehousename
        FROM warehouse.replenishment_outbound_gi
        WHERE customercode IS NOT NULL
          AND warehousecode IS NOT NULL
        """
    )

    scope_df = pd.read_sql(scope_sql, engine)
    source = "warehouse.replenishment_outbound_gi"

    if scope_df is None or scope_df.empty:
        return pd.DataFrame(columns=["customercode", "customername", "warehousecode", "warehousename"])

    # normalize key columns for stable filtering/joining
    scope_df["customercode"] = scope_df["customercode"].astype(str).str.strip()
    scope_df["warehousecode"] = scope_df["warehousecode"].astype(str).str.strip()
    scope_df["customername"] = scope_df["customername"].fillna("").astype(str)
    scope_df["warehousename"] = scope_df["warehousename"].fillna("").astype(str)

    scope_df = scope_df[
        (scope_df["customercode"] != "")
        & (scope_df["warehousecode"] != "")
    ].drop_duplicates(subset=["customercode", "warehousecode"]).reset_index(drop=True)

    print(f"[SCOPE] loaded {len(scope_df)} pairs from {source}")
    return scope_df


def _build_scope_pairs(
    customer_warehouses: Dict[str, List[str] | None] | None,
    scope_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """
    Resolve final (customercode, warehousecode) pairs to run.

    - If customer_warehouses is None: use all pairs from scope_df.
    - If customer_warehouses has explicit warehouses: use those pairs directly.
    - If customer_warehouses value is None/[]: expand from scope_df for that customer.
    """
    if scope_df is None:
        scope_df = pd.DataFrame(columns=["customercode", "warehousecode"])

    if customer_warehouses is None:
        return [
            (str(r.customercode), str(r.warehousecode))
            for r in scope_df[["customercode", "warehousecode"]].itertuples(index=False)
        ]

    out: list[tuple[str, str]] = []
    for customer_code, warehousecodes in customer_warehouses.items():
        cust = str(customer_code)

        if warehousecodes and len(warehousecodes) > 0:
            for wh in warehousecodes:
                if wh is None:
                    continue
                out.append((cust, str(wh)))
            continue

        # expand from DB scope for this customer
        customer_rows = scope_df[scope_df["customercode"].astype(str) == cust]
        if customer_rows.empty:
            print(f"[SCOPE] no warehouse found in scope for customer={cust} -> skip")
            continue
        out.extend(
            (cust, str(wh))
            for wh in customer_rows["warehousecode"].dropna().astype(str).unique().tolist()
            if str(wh).strip()
        )

    # unique preserve order
    seen = set()
    unique_out: list[tuple[str, str]] = []
    for pair in out:
        if pair in seen:
            continue
        seen.add(pair)
        unique_out.append(pair)
    return unique_out


def replace_replenishment_demand_forecast(df: pd.DataFrame):
    if df is None or df.empty:
        print("[SKIP] nothing to upload")
        return

    engine = db_manager.supabase_engine

    # เลือกคอลัมน์ให้ตรงกับ table (เรียง: ds, customercode, warehousecode, Shop_SKU, yhat)
    cols = ["ds", "customercode", "warehousecode", "Shop_SKU", "yhat"]
    upload = df[[c for c in cols if c in df.columns]].copy()
    upload["customercode"] = upload["customercode"].astype(str).str.strip()
    upload["warehousecode"] = upload["warehousecode"].where(
        upload["warehousecode"].notna(), None
    )
    upload["warehousecode"] = upload["warehousecode"].map(
        lambda x: str(x).strip() if x is not None else None
    )

    scope_pairs = (
        upload[["customercode", "warehousecode"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    with engine.begin() as conn:
        # Replace only scopes in this run (customer/warehouse), not whole table.
        for customercode, warehousecode in scope_pairs:
            if warehousecode is None or warehousecode == "":
                conn.execute(
                    text(
                        """
                        DELETE FROM warehouse.replenishment_demand_forecast
                        WHERE customercode = :cust
                          AND warehousecode IS NULL
                        """
                    ),
                    {"cust": customercode},
                )
            else:
                conn.execute(
                    text(
                        """
                        DELETE FROM warehouse.replenishment_demand_forecast
                        WHERE customercode = :cust
                          AND warehousecode = :wh
                        """
                    ),
                    {"cust": customercode, "wh": warehousecode},
                )
        upload.to_sql(
            "replenishment_demand_forecast",
            conn,
            schema="warehouse",
            if_exists="append",
            index=False,
            method="multi",
        )

    print(f"[DONE] replaced {len(df)} rows for scoped customer/warehouse pairs")


def resolve_training_scope_pairs(
    customer_warehouses: Dict[str, List[str] | None] | None = None,
    engine=None,
) -> list[tuple[str, str]]:
    """
    Resolve (customercode, warehousecode) pairs from Supabase scope + request mapping.
    """
    if engine is None:
        engine = db_manager.supabase_engine
    scope_df = load_training_scope_df(engine=engine)
    return _build_scope_pairs(customer_warehouses=customer_warehouses, scope_df=scope_df)


def _run_train_pair_with_frames(
    customer_code: str,
    warehousecode: str,
    oms_pandas_df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    config: dict | None,
    engine,
    *,
    periods: int = 14,
) -> Dict[str, Any]:
    """
    Train + upload forecast for one pair; replace replenishment_demand_forecast for that scope.
    On recoverable failure returns {"ok": False, ...} (does not raise).
    """
    try:
        db_config = load_demand_forecast_config_for(
            customercode=customer_code,
            warehousecode=warehousecode,
            engine=engine,
        )
        effective_config = _deep_merge_runtime(db_config, config)

        _, forecast_df_future = train_forecast_and_upload_models(
            customer_code=customer_code,
            oms_pandas_df=oms_pandas_df,
            holiday_df=holiday_df,
            promo_json_list=None,
            config=effective_config,
            periods=periods,
            promo_only=False,
            warehousecode=warehousecode,
        )

        upload_df = to_replenishment_forecast_upload_df(forecast_df_future)
        if upload_df is None or upload_df.empty:
            print(
                f"[TRAIN_PAIR] skip empty output for customer={customer_code}, warehouse={warehousecode}"
            )
            return {
                "ok": True,
                "customercode": customer_code,
                "warehousecode": warehousecode,
                "skipped": True,
                "reason": "empty_upload",
            }

        replace_replenishment_demand_forecast(upload_df)
        return {
            "ok": True,
            "customercode": customer_code,
            "warehousecode": warehousecode,
        }
    except Exception as e:
        print(f"[TRAIN_PAIR] customer={customer_code}, warehouse={warehousecode} failed: {e}")
        return {
            "ok": False,
            "customercode": customer_code,
            "warehousecode": warehousecode,
            "error": str(e),
        }


def train_single_pair(
    customer_code: str,
    warehousecode: str,
    config: dict | None = None,
    *,
    periods: int = 14,
) -> Dict[str, Any]:
    """
    Durable activity entrypoint: load OMS + holiday per invocation, train one (customer, warehouse).
    Does not raise on training failure; returns {"ok": bool, ...}.
    """
    engine = db_manager.supabase_engine
    oms_pandas_df = get_oms_pandas_df()
    holiday_df = get_holiday_df()
    return _run_train_pair_with_frames(
        customer_code,
        warehousecode,
        oms_pandas_df,
        holiday_df,
        config,
        engine,
        periods=periods,
    )


# def run_full_training(
#     *,
#     customer_code: str,
#     periods: int = 14,
#     warehousecodes: list[str] | None = None,   # ✅ ADD
# ):
#     """
#     Normal full run (no promo):
#     - load oms + holiday inside
#     - config=None (main will normalize default)
#     """
#     oms_pandas_df = get_oms_pandas_df()
#     holiday_df = get_holiday_df()

#     forecast_df_all, forecast_df_future = train_forecast_and_upload_models(
#         customer_code=customer_code,
#         oms_pandas_df=oms_pandas_df,
#         holiday_df=holiday_df,
#         promo_json_list=None,
#         config=None,          # ✅ main handles default
#         periods=periods,
#         promo_only=False,     # ✅ สำคัญ
#     )
#     return forecast_df_all, forecast_df_future

#######################################


# ---------------------------------------------------------------------------
# 6) Durable/batch entrypoint (วน customer + warehouse แล้วเขียนผลรวม)
# ---------------------------------------------------------------------------
def main_train_multi(
    customer_warehouses: Dict[str, List[str] | None] | None = None,
    config: dict | None = None,
) -> None:
    """
    รัน full training หลาย customer_code (CLI / local): โหลด OMS + holiday ครั้งเดียว แล้ววนคู่ละครั้ง
    เขียน replenishment_demand_forecast ต่อคู่ (ลบเฉพาะ scope คู่นั้นแล้ว append) เหมือนรวม concat ครั้งเดียว
    เมื่อคู่ไม่ซ้อนกัน

    Parameters
    ----------
    customer_warehouses:
        dict ที่ key = customer_code, value = list warehousecode หรือ None/[] (ไม่ใส่ = ให้ auto จาก OMS)

        - ใส่ warehouse: {\"CUST01\": [\"WH01\", \"WH02\"]}
        - ไม่ใส่ warehouse (auto): {\"CUST02\": None} หรือ {\"CUST02\": []}
    """
    engine = db_manager.supabase_engine
    scope_pairs = resolve_training_scope_pairs(
        customer_warehouses=customer_warehouses, engine=engine
    )

    if not scope_pairs:
        print("[MAIN_MULTI] no customer/warehouse pairs to run")
        return

    oms_pandas_df = get_oms_pandas_df()
    holiday_df = get_holiday_df()

    for customer_code, warehousecode in scope_pairs:
        print(f"[MAIN_MULTI] run customer={customer_code}, warehouse={warehousecode}")
        _run_train_pair_with_frames(
            customer_code,
            warehousecode,
            oms_pandas_df,
            holiday_df,
            config,
            engine,
            periods=14,
        )


if __name__ == "__main__":
    """
    NOTE debug/manual run (ไม่ผ่าน durable).

    Mapping ที่ fix ไว้ตอนนี้:
        {
          "0000000811": ["6140001663"],
          "0003056849": ["1130003312"],
          "0003056833": ["1130003312"],
          "0003057409": ["1130000015"],
        }

    วิธีรัน:
        python -m demand_forecast.train
    """

    hardcoded_customer_warehouses: Dict[str, List[str]] = {
        "0000000811": ["6140001663"],
        "0003056849": ["1130003312"],
        "0003056833": ["1130003312"],
        "0003057409": ["1130000015"]
    }

    print("[CLI] Running hardcoded main_train_multi with mapping:")
    for c, whs in hardcoded_customer_warehouses.items():
        print(f"  - customer={c}, warehouses={whs}")

    main_train_multi(hardcoded_customer_warehouses)
