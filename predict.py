"""
predict_promo_sku.py

Load Prophet model JSON from Supabase Storage (S3-compatible) and run prediction
for promo SKUs, ensuring promo_flag=1 is reflected on the promo dates.

Key idea:
- The trained model may include extra regressors (e.g., promo_flag).
- During prediction, you MUST supply those regressors in the future dataframe.
- This module builds a promo_daily_df from promo_json_list and merges promo_flag into future.

Expected promo_json_list format (INCLUDE):
[
  {
    "customer_code": "0000000811",
    "promotion_name": "...",
    "promotion_code": "...",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "apply_mode": "INCLUDE",
    "include_sku": [{"sku_code":"...", ...}, ...]
  }
]
"""
import os
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config
from prophet.serialize import model_from_json

# -----------------------------
# Supabase S3 client + model IO
# -----------------------------
def get_supabase_s3_client():
    supabase_url = os.getenv("SUPABASE_URL")
    region = os.getenv("SUPABASE_REGION", "ap-southeast-1")
    aws_key_id = os.getenv("SUPABASE_AWS_KEY_ID")
    aws_secret = os.getenv("SUPABASE_AWS_SECRET_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is not set")
    if not aws_key_id or not aws_secret:
        raise ValueError("SUPABASE_AWS_KEY_ID / SUPABASE_AWS_SECRET_KEY not set")

    # If SUPABASE_URL is base url -> append /storage/v1/s3
    endpoint_url = supabase_url if supabase_url.endswith("/storage/v1/s3") else f"{supabase_url}/storage/v1/s3"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=aws_key_id,
        aws_secret_access_key=aws_secret,
        config=Config(signature_version="s3v4"),
    )


def load_prophet_model_from_s3(*, customer_code: str, sku: str):
    bucket = os.getenv("SUPABASE_MODEL_BUCKET", "").strip()
    if not bucket:
        raise ValueError("SUPABASE_MODEL_BUCKET is not set")

    s3 = get_supabase_s3_client()
    key = f"{customer_code}/prophet/{sku}.json"
    print("BUCKET =", bucket)
    print("TRY KEY =", key)

    obj = s3.get_object(Bucket=bucket, Key=key)
    print("[S3] ETag =", obj.get("ETag"))
    print("[S3] LastModified =", obj.get("LastModified"))
    print("[S3] content-type =", obj.get("ContentType"))

    # ✅ read ONCE
    json_bytes = obj["Body"].read()
    print("[S3] bytes =", len(json_bytes))
    print("[S3] first 100 bytes =", json_bytes[:100])

    if not json_bytes:
        raise ValueError("Empty body from S3 (read returned 0 bytes).")

    json_str = json_bytes.decode("utf-8", errors="strict").lstrip()

    # กันเคสโดนส่งกลับเป็น HTML error page
    if json_str.startswith("<"):
        raise ValueError(f"S3 returned non-JSON (looks like HTML). First 200 chars: {json_str[:200]!r}")
    if not json_str.startswith("{"):
        raise ValueError(f"S3 returned non-JSON. First 200 chars: {json_str[:200]!r}")

    model = model_from_json(json_str)
    last_ds = pd.to_datetime(model.history["ds"]).max()
    print("[MODEL] history max ds =", last_ds)
    return model



# -----------------------------
# Promo helpers (promo_flag only)
# -----------------------------
def extract_skus_from_promo_json(promo_json_list: list) -> List[str]:
    """Return unique SKU codes appearing in INCLUDE lists."""
    skus = set()
    for p in promo_json_list or []:
        for s in p.get("include_sku", []) or []:
            code = s.get("sku_code")
            if code:
                skus.add(str(code))
    return sorted(skus)


def _normalize_promotions(promo_json_list: list) -> pd.DataFrame:
    """
    Normalize promo_json_list into a master table with at least:
      apply_mode, start_date, end_date, sku_code
    """
    rows = []
    for p in promo_json_list or []:
        apply_mode = (p.get("apply_mode", "ALL") or "ALL").upper()
        base = {
            "apply_mode": apply_mode,
            "start_date": p.get("start_date"),
            "end_date": p.get("end_date"),
        }

        if apply_mode == "ALL":
            rows.append({**base, "sku_code": None})
        elif apply_mode == "INCLUDE":
            for s in p.get("include_sku", []) or []:
                rows.append({**base, "sku_code": s.get("sku_code")})
        else:
            raise ValueError(f"Unknown apply_mode: {apply_mode}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.normalize()
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["start_date", "end_date"])
    df = df[df["end_date"] >= df["start_date"]].copy()
    df["sku_code"] = df["sku_code"].astype("object")
    return df.reset_index(drop=True)


def build_promo_daily_df(
    promo_json_list: list,
    all_skus_for_all_mode: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build daily promo flags:
      ds, Shop_SKU, promo_flag
    Notes:
    - If apply_mode=ALL exists, you must pass all_skus_for_all_mode.
    """
    master = _normalize_promotions(promo_json_list)
    if master.empty:
        return pd.DataFrame(columns=["ds", "Shop_SKU", "promo_flag"])

    need_all = (master["apply_mode"] == "ALL").any()
    if need_all and (not all_skus_for_all_mode):
        raise ValueError(
            "promo_json_list contains apply_mode='ALL' but all_skus_for_all_mode was not provided."
        )

    expanded = []
    for _, r in master.iterrows():
        if r["apply_mode"] == "ALL":
            target_skus = list(all_skus_for_all_mode or [])
        else:
            target_skus = [str(r["sku_code"])] if pd.notna(r["sku_code"]) else []

        for sku in target_skus:
            for d in pd.date_range(r["start_date"], r["end_date"], freq="D"):
                expanded.append({"ds": d, "Shop_SKU": sku, "promo_flag": 1})

    if not expanded:
        return pd.DataFrame(columns=["ds", "Shop_SKU", "promo_flag"])

    promo_daily_df = pd.DataFrame(expanded)
    promo_daily_df = (
        promo_daily_df.groupby(["ds", "Shop_SKU"], as_index=False)
        .agg(promo_flag=("promo_flag", "max"))
        .sort_values(["Shop_SKU", "ds"])
        .reset_index(drop=True)
    )
    return promo_daily_df


def _promo_daily_for_one_sku(promo_daily_df: pd.DataFrame, sku: str) -> pd.DataFrame:
    if promo_daily_df is None or promo_daily_df.empty:
        return pd.DataFrame(columns=["ds", "promo_flag"])
    out = promo_daily_df[promo_daily_df["Shop_SKU"].astype(str) == str(sku)][["ds", "promo_flag"]].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.normalize()
    out["promo_flag"] = out["promo_flag"].fillna(0).astype(int)
    return out


# -----------------------------
# Prediction with promo_flag
# -----------------------------
def predict_prophet(
    model,
    periods: int = 14,
    promo_daily_df_sku: Optional[pd.DataFrame] = None,
    clip_nonnegative: bool = True,
) -> pd.DataFrame:
    """
    Predict next N days (future-only) from the model's last training date,
    and inject required regressors (especially promo_flag).

    promo_daily_df_sku: columns ds, promo_flag (daily, for this sku)
    """
    future = model.make_future_dataframe(periods=periods, freq="D", include_history=False)
    future["ds"] = pd.to_datetime(future["ds"], errors="coerce").dt.normalize()

    # Fill extra regressors with 0 by default
    extra_regs = getattr(model, "extra_regressors", {}) or {}
    for reg in extra_regs.keys():
        future[reg] = 0

    # If the model has promo_flag regressor, merge promo days into future
    if "promo_flag" in extra_regs:
        if promo_daily_df_sku is not None and not promo_daily_df_sku.empty:
            tmp = promo_daily_df_sku.copy()
            tmp["ds"] = pd.to_datetime(tmp["ds"], errors="coerce").dt.normalize()
            tmp = tmp[["ds", "promo_flag"]].dropna(subset=["ds"])
            future = future.merge(tmp, on="ds", how="left", suffixes=("", "_p"))
            # If promo_flag exists twice, keep the merged one
            if "promo_flag_p" in future.columns:
                future["promo_flag"] = future["promo_flag_p"].fillna(future["promo_flag"]).fillna(0).astype(int)
                future.drop(columns=["promo_flag_p"], inplace=True)
            else:
                future["promo_flag"] = future["promo_flag"].fillna(0).astype(int)
        else:
            future["promo_flag"] = 0

    forecast = model.predict(future)
    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce").dt.normalize()

    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    if clip_nonnegative:
        for c in ["yhat", "yhat_lower", "yhat_upper"]:
            out[c] = (
                out[c]
                .fillna(0)
                .replace([np.inf, -np.inf], 0)
                .clip(lower=0)
                .round()
                .astype(int)
            )

    return out.sort_values("ds").reset_index(drop=True)


def load_and_predict_prophet(
    *,
    customer_code: str,
    sku: str,
    periods: int = 14,
    promo_json_list: Optional[list] = None,
    all_skus_for_all_mode: Optional[List[str]] = None,
    clip_nonnegative: bool = True,
) -> pd.DataFrame:
    """
    Load model from Supabase and predict.
    If promo_json_list is provided, promo_flag will be set to 1 on promo dates.
    """
    model = load_prophet_model_from_s3(customer_code=customer_code, sku=sku)

    promo_daily_df = None
    promo_daily_df_sku = None
    if promo_json_list:
        promo_daily_df = build_promo_daily_df(promo_json_list, all_skus_for_all_mode=all_skus_for_all_mode)
        promo_daily_df_sku = _promo_daily_for_one_sku(promo_daily_df, sku)

    fcst = predict_prophet(
        model,
        periods=periods,
        promo_daily_df_sku=promo_daily_df_sku,
        clip_nonnegative=clip_nonnegative,
    )

    fcst.insert(1, "Shop_SKU", str(sku))

    # Add promo_flag column to output (handy for debugging)
    if promo_daily_df_sku is not None and not promo_daily_df_sku.empty:
        out = fcst.merge(promo_daily_df_sku, on="ds", how="left")
        out["promo_flag"] = out["promo_flag"].fillna(0).astype(int)
    else:
        out = fcst.copy()
        out["promo_flag"] = 0

    return out.sort_values(["Shop_SKU", "ds"]).reset_index(drop=True)


# -----------------------------
# Runner: promo_json_list -> predict all promo SKUs
# -----------------------------
def predict_from_promo_json_list(
    *,
    customer_code: str,
    promo_json_list: list,
    periods: int = 14,
    all_skus_for_all_mode: Optional[List[str]] = None, 
) -> pd.DataFrame:
    promo_skus = extract_skus_from_promo_json(promo_json_list)
    if not promo_skus:
        raise ValueError("No SKU found in promo_json_list['include_sku'].")

    promo_daily_df = build_promo_daily_df(
        promo_json_list, all_skus_for_all_mode=all_skus_for_all_mode
    )
    promo_daily_map = {sku: _promo_daily_for_one_sku(promo_daily_df, sku) for sku in promo_skus}

    outputs = []
    for sku in promo_skus:
        model = load_prophet_model_from_s3(customer_code=customer_code, sku=sku)

        out = predict_prophet(
            model,
            periods=periods,
            promo_daily_df_sku=promo_daily_map.get(sku),
            clip_nonnegative=True,
        )

        # ✅ stamp customer + sku
        out.insert(1, "customer_code", customer_code)
        out.insert(2, "Shop_SKU", str(sku))

        # ✅ attach promo_flag
        pdf = promo_daily_map.get(sku)
        if pdf is not None and not pdf.empty:
            out = out.merge(pdf, on="ds", how="left")
            out["promo_flag"] = out["promo_flag"].fillna(0).astype(int)
        else:
            out["promo_flag"] = 0

        outputs.append(out)

    return (
        pd.concat(outputs, ignore_index=True)
          .sort_values(["Shop_SKU", "ds"])
          .reset_index(drop=True)
    )



def generate_predict_commands(
    *,
    customer_code: str,
    promo_json_list: list,
    periods: int = 14,
) -> List[str]:
    """
    Utility: return a list of Python call-strings (commands) to run prediction per SKU.
    Useful if you want to submit as separate jobs / logs.
    """
    promo_skus = extract_skus_from_promo_json(promo_json_list)
    return [
        f"load_and_predict_prophet(customer_code='{customer_code}', sku='{sku}', periods={periods}, promo_json_list=promo_json_list)"
        for sku in promo_skus
    ]


if __name__ == "__main__":
    # Example usage
    promo_json_list = [
        {
            "customer_code": "0000000811",
            "promotion_name": "payday test",
            "promotion_code": "payday_test",
            "start_date": "2026-02-03",
            "end_date": "2026-02-07",
            "apply_mode": "INCLUDE",
            "include_sku": [
                {"sku_code": "Z220C11010X0000010", "discount_amount": 0, "discount_type": "fixed"},
                {"sku_code": "Z220SL900550001010", "discount_amount": 0, "discount_type": "percent"},
                {"sku_code": "Z220C11003H0400010", "discount_amount": 0, "discount_type": "fixed"},
                {"sku_code": "Z236CT6403Z2PHMX18", "discount_amount": 0, "discount_type": "fixed"},
            ],
        }
    ]

    # 1) Generate commands (optional)
    cmds = generate_predict_commands(customer_code="0000000811", promo_json_list=promo_json_list, periods=14)
    print("Commands:")
    for c in cmds:
        print(" -", c)

    # 2) Predict all promo SKUs (single dataframe)
    pred_all = predict_from_promo_json_list(
        customer_code="0000000811",
        promo_json_list=promo_json_list,
        periods=14,
    )
    print(pred_all)
