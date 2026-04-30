"""
Prepare canonical demand input for demand forecast.

Output schema from `main()`:
- ds
- customer_code
- warehousecode
- Shop_SKU
- y

Source split:
- GI table: all customers except COTTO
- ORDER table: only COTTO
"""

import pandas as pd
from sqlalchemy import text

from .database_engine import db_manager
from .util.DataAdapter import DataAdapter

COTTO_CUSTOMER_CODE = "0000000811"
SUPABASE_ORDER_TABLE = "warehouse.replenishment_order"
SUPABASE_GI_LOG_TABLE = "warehouse.replenishment_outbound_gi_log"
WAREHOUSE_CODE_CANDIDATES = [
    "warehousecode",
    "warehouse_code",
    "WarehouseCode",
    "WH_CODE",
]

SKU_MAPPING_BY_CUSTOMER = {
    "0000000811": {
        "Z236CT683AXHMXXX18": "Z236CT683AXHMTGX18",
        "Z236CT683HMXXXXX18": "Z236CT683HMTGXXX18",
        "Z236CT670VHMXXXX18": "Z236CT670VHMTGXX18",
        "Z236S290XXXXXXXX18": "Z236S290TGXXXXXX18",
        "Z234S17HMXXXXXXX11": "Z234S17HMTGXXXXX11",
        "Z236CT673HMXXXXX18": "Z236CT673HMTGXXX18",
        "Z236CT666NWHMXXX16": "Z236CT666NWHMTGX16",
        "Z235CT709HMXXXXX18": "Z235CT709HMTGXXX18",
        "Z236CT680AXHMXXX18": "Z236CT680AXHMTGX18",
        "Z231CT1700HMXXXX11": "Z231CT1700HMTGXX11",
        "Z220C11003H0000010": "Z220C11003H0400010",
        "Z236CT665HMXXXXX18": "Z236CT665HMTGXXX18",
        "Z236CT665NHMXXXX18": "Z236CT665NHMTGXX18",
        "Z227V00181XWH11000": "Z227V00181NWH01000",
        "Z227V00181XBL11000": "Z227V00181NBL01000",
        "Z227V00181XRO11000": "Z227V00181NRO01000",
        "Z220C13512X0000510": "Z220C13512X1100510",
        "Z220C10192X0000510": "Z220C10192X0100510",
        "Z220C9208XX0001010": "Z220C9208XX0201010",
        "Z220C9205XX0001010": "Z220C9205XX0101010",
        "Z220C9377XX0009000": "Z220C9377XX0209000",
        "Z220C9206XX0001010": "Z236MF6011HMXXXX11",
        "Z220C12531X1100510": "Z220C12531X2200510",
        "Z220C13440X0000010": "Z220C13440X1000010",
        "Z231CT1161XXXXXX11": "Z231CT1161ANXXXX11",
        "Z220C1032XX0300510": "Z220C1032XX0700510",
    },
}


def _debug_y_summary(label: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print(f"[DEBUG] {label}: empty df")
        return
    y_series = pd.to_numeric(df.get("y"), errors="coerce").fillna(0.0)
    nonzero = int((y_series > 0).sum())
    print(
        f"[DEBUG] {label}: rows={len(df)} sum_y={float(y_series.sum()):.2f} "
        f"nonzero_y_rows={nonzero}"
    )
    if "customer_code" in df.columns and "warehousecode" in df.columns:
        top_pairs = (
            df.assign(_y=y_series)
            .groupby(["customer_code", "warehousecode"], dropna=False)["_y"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        print(f"[DEBUG] {label} top customer/warehouse sum_y:")
        print(top_pairs)


def extract_from_supabase(query: str, params=None, chunksize: int | None = None) -> pd.DataFrame:
    """Read query result from Supabase in chunks."""
    chunks = []
    for chunk in pd.read_sql(
        text(query),
        con=db_manager.supabase_engine,
        params=params,
        chunksize=chunksize,
    ):
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def rename_sku_by_customer(
    df: pd.DataFrame,
    mapping_by_customer: dict[str, dict[str, str]],
    customer_col: str = "customer_code",
    sku_col: str = "Shop_SKU",
) -> pd.DataFrame:
    """Apply customer-specific SKU remapping rules."""
    out = df.copy()
    for customer_code, mapping in mapping_by_customer.items():
        mask = out[customer_col].astype(str) == str(customer_code)
        if mask.any():
            out.loc[mask, sku_col] = out.loc[mask, sku_col].replace(mapping)
    return out


def apply_cotto_logic(df: pd.DataFrame) -> pd.DataFrame:
    """COTTO-only cleanup: rename SKU + keep valid payment rows after cutoff."""
    filter_date = pd.to_datetime("2025-02-02")
    out = rename_sku_by_customer(df, SKU_MAPPING_BY_CUSTOMER)
    return out[(out["ds"] >= filter_date) & (out["paymentdate"].notnull())]


def transform_source_df(df: pd.DataFrame, *, table_type: str) -> pd.DataFrame:
    """Normalize one source (gi/oms_order) to common column format."""
    if df is None or df.empty:
        return pd.DataFrame()

    print(f"[DEBUG] transform_source_df start table_type={table_type} rows={len(df)}")
    adapter = DataAdapter()
    out = adapter.adapter_table(df, table_type)
    out = out.loc[:, ~out.columns.duplicated()]

    # Normalize warehouse code into one canonical column.
    if "warehousecode" not in out.columns:
        for col in WAREHOUSE_CODE_CANDIDATES:
            if col in out.columns:
                out["warehousecode"] = out[col]
                break
    if "warehousecode" not in out.columns:
        out["warehousecode"] = None

    out = out[out["Shop_SKU"].notnull()].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.normalize()
    out["customer_code"] = out["customer_code"].astype(str).str.strip()
    out["Shop_SKU"] = out["Shop_SKU"].astype(str).str.strip()
    out["warehousecode"] = out["warehousecode"].astype(str).str.strip()
    out["y"] = pd.to_numeric(out["y"], errors="coerce").fillna(0.0)
    _debug_y_summary(f"{table_type} after adapt", out)

    yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    out = out[out["ds"].notna() & (out["ds"] <= yesterday)].copy()
    if out.empty:
        print(f"[DEBUG] transform_source_df table_type={table_type} empty after date filter")
        return out

    pieces = []
    for customer_code, group in out.groupby("customer_code"):
        # Apply special handling only for COTTO.
        if str(customer_code) == COTTO_CUSTOMER_CODE:
            pieces.append(apply_cotto_logic(group))
        else:
            pieces.append(group)

    if not pieces:
        return pd.DataFrame()
    merged = pd.concat(pieces, ignore_index=True)
    _debug_y_summary(f"{table_type} after per-customer logic", merged)
    return merged


def aggregate_daily_demand(transformed_df: pd.DataFrame, start_date: str = "2025-02-02") -> pd.DataFrame:
    """Aggregate to daily demand and fill missing date/SKU combinations with 0."""
    if transformed_df is None or transformed_df.empty:
        return pd.DataFrame(columns=["ds", "customer_code", "warehousecode", "Shop_SKU", "y"])

    _debug_y_summary("aggregate input", transformed_df)
    sum_df = transformed_df.groupby(["ds", "customer_code", "warehousecode", "Shop_SKU"])["y"].sum().reset_index()
    _debug_y_summary("aggregate grouped sum_df", sum_df)
    all_dates = pd.date_range(start=start_date, end=transformed_df["ds"].max(), name="ds").to_frame(index=False)
    all_skus = sum_df[["customer_code", "warehousecode", "Shop_SKU"]].drop_duplicates()

    final_df = (
        all_dates.merge(all_skus, how="cross")
        .merge(sum_df, on=["ds", "customer_code", "warehousecode", "Shop_SKU"], how="left")
        .fillna({"y": 0})
    )
    final_df["y"] = final_df["y"].astype(float)
    _debug_y_summary("aggregate final_df", final_df)
    return final_df


def prep_demandforecast_data() -> pd.DataFrame:

    # 1) GI path: everyone except COTTO
    gi_df = extract_from_supabase(
        f"""
        SELECT *
        FROM {SUPABASE_GI_LOG_TABLE}
        WHERE customercode::text <> '{COTTO_CUSTOMER_CODE}'
        """,
        chunksize=10_000,
    )
    print(f"[DEBUG] GI raw rows={len(gi_df)}")
    transformed_gi_df = transform_source_df(gi_df, table_type="gi")

    # 2) ORDER path: COTTO only
    try:
        order_df = extract_from_supabase(
            f"""
            SELECT *
            FROM {SUPABASE_ORDER_TABLE}
            WHERE sendercode::text = '{COTTO_CUSTOMER_CODE}'
            """,
            chunksize=10_000,
        )
        print(f"[DEBUG] ORDER raw rows={len(order_df)}")
        transformed_order_df = transform_source_df(order_df, table_type="oms_order")
    except Exception as exc:
        print(f"[WARN] ORDER source unavailable, skip COTTO order prep: {exc}")
        transformed_order_df = pd.DataFrame()

    # 3) Merge both paths and build canonical daily demand
    transformed_frames = [df for df in [transformed_gi_df, transformed_order_df] if df is not None and not df.empty]
    if not transformed_frames:
        print("[WARN] No transformed rows from both GI/ORDER sources")
        return pd.DataFrame(columns=["ds", "customer_code", "warehousecode", "Shop_SKU", "y"])

    transformed_df = pd.concat(transformed_frames, ignore_index=True)
    print(f"[INFO] Transformed merged df: {len(transformed_df)} rows")

    final_df = aggregate_daily_demand(transformed_df)
    return final_df

if __name__ == "__main__":
    prepared_df = prep_demandforecast_data()
    print(f"[INFO] Prepared rows: {len(prepared_df)}")
    size_mb = prepared_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"[INFO] Prepared dataframe size: {size_mb:.2f} MB")
    print(prepared_df.head(5))
    print(prepared_df[(prepared_df["customer_code"]=="0000000811") & (prepared_df["warehousecode"]=="6140001663")]["y"].sum())
