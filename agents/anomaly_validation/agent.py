from __future__ import annotations
from .data import load_dataset, select_history
from .models import ValidationConfig
from .validators import data_quality_checks, historical_anomaly_summary, forecast_checks, inventory_risk

AGENT_NAME="anomaly_validation"
CAPABILITIES=["forecast_validation","anomaly_detection","data_quality_validation","stockout_risk_validation","overstock_risk_validation"]

def _severity(issues):
    levels={x.get("severity") for x in issues}
    return "fail" if "high" in levels else ("warning" if "medium" in levels else "pass")

def anomaly_validation_agent(state: dict, dataset_path=None, config: ValidationConfig | None=None) -> dict:
    """Phase-2 Anomaly/Validation Agent. Reads shared state; never invokes other agents."""
    cfg=config or ValidationConfig()
    if not isinstance(state,dict): raise TypeError("state must be a dictionary")
    request=state.get("request") or {}
    results=state.get("agent_results") or {}
    forecast=results.get("demand_forecasting") or {}
    data_result=results.get("data_analyst") or {}
    path=dataset_path or request.get("dataset_path")
    try:
        history=select_history(load_dataset(path), request)
        dq=data_quality_checks(history,cfg)
        hist=historical_anomaly_summary(history,cfg)
    except Exception as e:
        history=None
        dq=[{"code":"DATASET_ERROR","severity":"high","message":str(e)}]
        hist={"method":"robust_zscore","count":0,"rate":0.0,"recent_anomaly":False,"status":"unavailable"}
    fissues, fmetrics=forecast_checks(forecast, history if history is not None else __import__('pandas').DataFrame({"quantity":[]}), cfg)
    risk=inventory_risk(data_result,forecast,cfg)
    issues=dq+fissues
    if hist.get("recent_anomaly"):
        issues.append({"code":"RECENT_DEMAND_ANOMALY","severity":"medium","message":"Most recent historical demand is anomalous versus the SKU history."})
    if risk["stockout_risk"]=="high": issues.append({"code":"STOCKOUT_RISK","severity":"high","message":"Current stock is materially below forecast demand."})
    if risk["overstock_risk"]=="high": issues.append({"code":"OVERSTOCK_RISK","severity":"medium","message":"Current stock is materially above forecast demand."})
    status=_severity(issues)
    reforecast=any(i["code"] in {"INVALID_FORECAST","NEGATIVE_FORECAST","FORECAST_DEVIATION","DATASET_ERROR"} and i["severity"]=="high" for i in issues)
    return {
        "analysis_type":"forecast_validation",
        "restaurant_id":request.get("restaurant_id") or forecast.get("restaurant_id") or data_result.get("restaurant_id"),
        "restaurant_name":request.get("restaurant_name") or forecast.get("restaurant_name") or data_result.get("restaurant_name"),
        "menu_item_id":request.get("menu_item_id") or forecast.get("menu_item_id") or data_result.get("menu_item_id"),
        "menu_item_name":request.get("menu_item_name") or forecast.get("menu_item_name") or data_result.get("menu_item_name"),
        "validation_status":status,
        "forecast_valid":status!="fail",
        "reforecast_recommended":reforecast,
        "forecast_metrics":fmetrics,
        "historical_anomalies":hist,
        "inventory_risk":risk,
        "issues":issues,
        "agent_name":AGENT_NAME,
        "capabilities":CAPABILITIES,
    }

def agent(state: dict) -> dict:
    """Integration alias matching the Phase-1 preferred def agent(state) contract."""
    return anomaly_validation_agent(state)
