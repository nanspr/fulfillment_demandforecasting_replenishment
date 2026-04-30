from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping



def _deep_merge_dict(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_positive_int_list(values: Any, fallback: list[int]) -> list[int]:
    if values is None:
        return fallback

    out: list[int] = []
    for value in values:
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            continue
        if int_value > 0 and int_value not in out:
            out.append(int_value)
    return out or fallback


def _coerce_float_list(values: Any, fallback: list[float]) -> list[float]:
    if values is None:
        return fallback

    out: list[float] = []
    for value in values:
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            continue
        out.append(float_value)
    return out or fallback


@dataclass
class CustomerScopeConfig:
    customer_code: str = ""
    warehousecode: str | None = None
    warehousecodes: list[str] | None = None
    periods: int = 14
    promo_only: bool = False


@dataclass
class HolidayConfig:
    use_doubleday_peak: bool = True
    use_public_holiday: bool = True
    use_midmonth_dates: bool = True
    use_payday_dates:   bool = True


@dataclass
class ImpactDateAdvancedConfig:
    holiday_lower_window: int = 0
    holiday_upper_window: int = 1
    top_skus_n: int = 30
    cv_bands: list[float] = field(default_factory=lambda: [0.2, 0.6])
    zscore_rules: list[dict[str, float]] = field(
        default_factory=lambda: [
            {"window": 10, "threshold": 2.5},
            {"window": 7, "threshold": 2.0},
            {"window": 5, "threshold": 1.5},
        ]
    )


@dataclass
class LagFeatureConfig:
    lag_days: list[int] = field(default_factory=lambda: [1, 7])
    rolling_mean_windows: list[int] = field(default_factory=lambda: [3])


@dataclass
class PromoConfig:
    use_promotions: bool = True
    holiday_lower_window: int = 0
    holiday_upper_window: int = 0


@dataclass
class ProphetConfig:
    weekly_seasonality: bool = True
    daily_seasonality: bool = True
    seasonality_mode: str = "multiplicative"
    changepoint_prior_scale: float = 0.5
    changepoint_range: float = 0.1
    seasonality_prior_scale: float = 5.0


@dataclass
class ModelSelectionConfig:
    prophet_volume_threshold: float = 0.8
    prophet_skus_override: list[str] | None = None


@dataclass
class AdvancedConfig:
    impact_dates: ImpactDateAdvancedConfig = field(default_factory=ImpactDateAdvancedConfig)
    prophet: ProphetConfig = field(default_factory=ProphetConfig)
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)


@dataclass
class ExecutionConfig:
    test_days: int = 30
    batch_size: int = 50
    n_jobs: int = 8


@dataclass
class ForecastTrainingConfig:
    customer: CustomerScopeConfig = field(default_factory=CustomerScopeConfig)
    holiday: HolidayConfig = field(default_factory=HolidayConfig)
    lag_features: LagFeatureConfig = field(default_factory=LagFeatureConfig)
    promo: PromoConfig = field(default_factory=PromoConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None = None) -> "ForecastTrainingConfig":
        payload = dict(payload or {})
        advanced_payload = dict(payload.get("advanced") or {})

        # Backward compatibility for the previous top-level structure.
        if "prophet" in payload and "prophet" not in advanced_payload:
            advanced_payload["prophet"] = payload.get("prophet")
        if "model_selection" in payload and "model_selection" not in advanced_payload:
            advanced_payload["model_selection"] = payload.get("model_selection")
        if "impact_dates" in payload and "impact_dates" not in advanced_payload:
            impact_payload = payload.get("impact_dates") or {}
            if isinstance(impact_payload, Mapping):
                advanced_payload["impact_dates"] = impact_payload

        return cls(
            customer=_build_dataclass(CustomerScopeConfig, payload.get("customer")),
            holiday=_build_dataclass(HolidayConfig, payload.get("holiday")),
            lag_features=_build_dataclass(LagFeatureConfig, payload.get("lag_features")),
            promo=_build_dataclass(PromoConfig, payload.get("promo")),
            advanced=_build_advanced_config(advanced_payload),
            execution=_build_dataclass(ExecutionConfig, payload.get("execution")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["lag_features"]["lag_days"] = _coerce_positive_int_list(
            data["lag_features"].get("lag_days"),
            [1, 7],
        )
        data["lag_features"]["rolling_mean_windows"] = _coerce_positive_int_list(
            data["lag_features"].get("rolling_mean_windows"),
            [3],
        )
        data["advanced"]["impact_dates"]["top_skus_n"] = max(
            1, int(data["advanced"]["impact_dates"].get("top_skus_n", 30))
        )
        data["advanced"]["impact_dates"]["cv_bands"] = _coerce_float_list(
            data["advanced"]["impact_dates"].get("cv_bands"),
            [0.2, 0.6],
        )
        data["advanced"]["impact_dates"]["zscore_rules"] = _normalize_zscore_rules(
            data["advanced"]["impact_dates"].get("zscore_rules")
        )
        data["advanced"]["model_selection"]["prophet_volume_threshold"] = min(
            1.0,
            max(0.0, float(data["advanced"]["model_selection"].get("prophet_volume_threshold", 0.8))),
        )

        return data

    def to_runtime_config(self) -> dict[str, Any]:
        data = self.to_dict()
        return {
            "customer": data["customer"],
            "holiday": data["holiday"],
            "impact_dates": {
                "lower_window": int(data["advanced"]["impact_dates"]["holiday_lower_window"]),
                "upper_window": int(data["advanced"]["impact_dates"]["holiday_upper_window"]),
                "top_skus_n": int(data["advanced"]["impact_dates"]["top_skus_n"]),
                "cv_bands": data["advanced"]["impact_dates"]["cv_bands"],
                "zscore_rules": data["advanced"]["impact_dates"]["zscore_rules"],
            },
            "lag_features": data["lag_features"],
            "promo": {
                "use_promotions": bool(data["promo"]["use_promotions"]),
                "lower_window": int(data["promo"]["holiday_lower_window"]),
                "upper_window": int(data["promo"]["holiday_upper_window"]),
            },
            "advanced": data["advanced"],
            "execution": data["execution"],
        }


def _build_dataclass(cls: type, payload: Mapping[str, Any] | None):
    default_obj = cls()
    default_data = asdict(default_obj)
    merged = _deep_merge_dict(default_data, payload or {})
    allowed_keys = {f.name for f in fields(cls)}
    filtered = {key: value for key, value in merged.items() if key in allowed_keys}
    return cls(**filtered)


def _build_advanced_config(payload: Mapping[str, Any] | None):
    payload = dict(payload or {})
    return AdvancedConfig(
        impact_dates=_build_dataclass(ImpactDateAdvancedConfig, payload.get("impact_dates")),
        prophet=_build_dataclass(ProphetConfig, payload.get("prophet")),
        model_selection=_build_dataclass(ModelSelectionConfig, payload.get("model_selection")),
    )


def _normalize_zscore_rules(values: Any) -> list[dict[str, float]]:
    fallback = [
        {"window": 10, "threshold": 2.5},
        {"window": 7, "threshold": 2.0},
        {"window": 5, "threshold": 1.5},
    ]
    if values is None:
        return fallback

    out: list[dict[str, float]] = []
    for rule in values:
        if not isinstance(rule, Mapping):
            continue
        try:
            window = int(rule.get("window"))
            threshold = float(rule.get("threshold"))
        except (TypeError, ValueError):
            continue
        if window > 0:
            out.append({"window": window, "threshold": threshold})

    return out or fallback


def normalize_training_config(
    config: ForecastTrainingConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(config, ForecastTrainingConfig):
        return config.to_runtime_config()
    if is_dataclass(config):
        return ForecastTrainingConfig.from_dict(asdict(config)).to_runtime_config()
    if isinstance(config, Mapping):
        return ForecastTrainingConfig.from_dict(config).to_runtime_config()
    return ForecastTrainingConfig().to_runtime_config()

