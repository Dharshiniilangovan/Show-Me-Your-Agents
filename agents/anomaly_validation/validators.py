from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .models import ValidationConfig

def _finite(v):
    try: return math.isfinite(float(v))
    except (TypeError, ValueError): return False

def data_quality_checks(history: pd.DataFrame, cfg: ValidationConfig) -> list[dict]:
    issues=[]
    if history.empty:
        return [{"code":"NO_HISTORY","severity":"high","message":"No matching historical rows were found."}]
    for c in ["date","quantity"]:
        rate=float(history[c].isna().mean())
        if rate > cfg.max_missing_rate:
            issues.append({"code":"MISSING_VALUES","severity":"high","field":c,"rate":round(rate,4),"message":f"{c} missing rate exceeds threshold."})
        elif rate > 0:
            issues.append({"code":"MISSING_VALUES","severity":"medium","field":c,"rate":round(rate,4),"message":f"{c} contains missing values."})
    neg=int((history["quantity"].fillna(0)<0).sum())
    if neg:
        issues.append({"code":"NEGATIVE_QUANTITY","severity":"high","count":neg,"message":"Negative demand values detected."})
    dups=int(history.duplicated(subset=["date","restaurant_id","menu_item_id"]).sum())
    if dups:
        issues.append({"code":"DUPLICATE_KEYS","severity":"medium","count":dups,"message":"Duplicate date/restaurant/item rows detected."})
    return issues

def historical_anomaly_summary(history: pd.DataFrame, cfg: ValidationConfig) -> dict:
    q=history["quantity"].dropna().astype(float)
    if len(q)<5:
        return {"method":"robust_zscore","count":0,"rate":0.0,"recent_anomaly":False,"status":"insufficient_history"}
    med=float(q.median()); mad=float(np.median(np.abs(q-med)))
    if mad == 0:
        flags=np.abs(q-med)>0
        scores=np.where(flags, np.inf, 0.0)
    else:
        scores=0.6745*(q-med)/mad
        flags=np.abs(scores)>cfg.robust_z_threshold
    count=int(np.sum(flags)); rate=float(count/len(q))
    recent=False
    if len(q):
        last=float(q.iloc[-1]); recent=bool(abs(0.6745*(last-med)/mad)>cfg.robust_z_threshold) if mad else bool(last!=med)
    return {"method":"robust_zscore","threshold":cfg.robust_z_threshold,"count":count,"rate":round(rate,4),"recent_anomaly":recent,"median":round(med,3),"mad":round(mad,3),"status":"ok"}

def forecast_checks(forecast: dict, history: pd.DataFrame, cfg: ValidationConfig) -> tuple[list[dict], dict]:
    issues=[]
    pred=forecast.get("predicted_demand")
    horizon=forecast.get("forecast_horizon")
    conf=forecast.get("confidence")
    if not _finite(pred):
        issues.append({"code":"INVALID_FORECAST","severity":"high","message":"predicted_demand is missing or non-numeric."})
        return issues, {"predicted_demand":pred,"historical_baseline":None,"deviation_pct":None}
    pred=float(pred)
    if pred<0: issues.append({"code":"NEGATIVE_FORECAST","severity":"high","message":"Forecast demand cannot be negative."})
    if horizon is None or not _finite(horizon) or float(horizon)<=0:
        issues.append({"code":"INVALID_HORIZON","severity":"high","message":"forecast_horizon must be positive."})
        horizon=1
    horizon=int(float(horizon))
    if conf is not None and _finite(conf) and float(conf)<cfg.low_confidence_threshold:
        issues.append({"code":"LOW_CONFIDENCE","severity":"medium","value":float(conf),"message":"Forecast confidence is below configured threshold."})
    q=history["quantity"].dropna().astype(float)
    baseline=None; deviation=None
    if len(q)>=cfg.min_history_points:
        daily=float(q.tail(min(56,len(q))).median())
        baseline=daily*horizon
        if baseline>0:
            deviation=(pred-baseline)/baseline*100.0
            a=abs(deviation)
            if a>=cfg.forecast_deviation_fail_pct:
                issues.append({"code":"FORECAST_DEVIATION","severity":"high","deviation_pct":round(deviation,2),"message":"Forecast is far from the recent historical baseline."})
            elif a>=cfg.forecast_deviation_warn_pct:
                issues.append({"code":"FORECAST_DEVIATION","severity":"medium","deviation_pct":round(deviation,2),"message":"Forecast materially differs from the recent historical baseline."})
    return issues, {"predicted_demand":round(pred,3),"historical_baseline":None if baseline is None else round(baseline,3),"deviation_pct":None if deviation is None else round(deviation,2)}

def inventory_risk(data_result: dict, forecast: dict, cfg: ValidationConfig) -> dict:
    stock=data_result.get("current_stock")
    pred=forecast.get("predicted_demand")
    if not (_finite(stock) and _finite(pred)):
        return {"stockout_risk":"unknown","overstock_risk":"unknown","current_stock":stock,"predicted_demand":pred,"coverage_ratio":None}
    stock=float(stock); pred=max(float(pred),0.0)
    ratio=stock/pred if pred>0 else (float("inf") if stock>0 else 1.0)
    if ratio<cfg.stockout_ratio_high: so="high"
    elif ratio<cfg.stockout_ratio_medium: so="medium"
    else: so="low"
    if ratio>=cfg.overstock_ratio_high: ov="high"
    elif ratio>=cfg.overstock_ratio_medium: ov="medium"
    else: ov="low"
    return {"stockout_risk":so,"overstock_risk":ov,"current_stock":stock,"predicted_demand":pred,"coverage_ratio":None if math.isinf(ratio) else round(ratio,3)}
