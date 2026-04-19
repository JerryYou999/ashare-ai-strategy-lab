from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "double_ma",
        "name": "双均线趋势",
        "category": "趋势",
        "auto_backtest": True,
        "description": "短期均线上穿长期均线买入，下穿卖出，适合趋势较明确的行情。",
        "default_params": {
            "short_window": 10,
            "long_window": 30,
            "position_size": 0.95,
        },
    },
    {
        "id": "bollinger_mean_reversion",
        "name": "布林带均值回归",
        "category": "均值回归",
        "auto_backtest": True,
        "description": "价格跌破下轨后尝试做多，回到中轨附近止盈，适合震荡市场。",
        "default_params": {
            "window": 20,
            "num_std": 2.0,
            "position_size": 0.90,
        },
    },
    {
        "id": "alexander_filter",
        "name": "Alexander Filter 突破",
        "category": "突破",
        "auto_backtest": True,
        "description": "价格从低点上行超过过滤阈值时做多，从高点回落超过阈值时离场。",
        "default_params": {
            "filter_pct": 0.05,
            "position_size": 0.95,
        },
    },
    {
        "id": "momentum_short_term",
        "name": "短期动量",
        "category": "动量",
        "auto_backtest": True,
        "description": "观察过去 N 天收益，若为正则继续顺势做多。",
        "default_params": {
            "lookback": 20,
            "entry_threshold": 0.03,
            "position_size": 0.95,
        },
    },
    {
        "id": "contrarian_long_term",
        "name": "长期反转",
        "category": "反转",
        "auto_backtest": True,
        "description": "观察较长期表现，若超跌则布局反转机会。",
        "default_params": {
            "lookback": 120,
            "entry_threshold": -0.15,
            "exit_threshold": 0.05,
            "position_size": 0.90,
        },
    },
    {
        "id": "gap_strategy",
        "name": "跳空策略",
        "category": "事件/开盘",
        "auto_backtest": True,
        "description": "基于今日开盘相对昨收的跳空幅度，做延续或反转判断。",
        "default_params": {
            "gap_threshold": 0.02,
            "mode": "reversal",
            "position_size": 0.90,
        },
    },
    {
        "id": "seasonal_halloween",
        "name": "季节效应",
        "category": "季节",
        "auto_backtest": True,
        "description": "按月份持有，演示季节性市场效应，可自定义持仓月份。",
        "default_params": {
            "hold_months": [11, 12, 1, 2, 3, 4],
            "position_size": 1.0,
        },
    },
    {
        "id": "news_event_beta",
        "name": "新闻事件漂移（Beta）",
        "category": "事件驱动",
        "auto_backtest": False,
        "description": "结合新闻和事件强度形成信号，首版用于研究展示，不默认自动回测。",
        "default_params": {
            "event_threshold": 0.6,
            "position_size": 0.80,
        },
    },
]


def list_templates() -> List[Dict[str, Any]]:
    return deepcopy(TEMPLATES)


def get_template(template_id: str) -> Dict[str, Any]:
    for item in TEMPLATES:
        if item["id"] == template_id:
            return deepcopy(item)
    raise KeyError(f"Unknown template id: {template_id}")


def strategy_schema() -> Dict[str, Any]:
    return {
        "template_id": "one of the supported templates",
        "stock_code": "e.g. 600519",
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "initial_capital": 100000,
        "position_size": 0.95,
        "params": {"template_specific_key": "template_specific_value"},
        "rationale": "short explanation",
        "risk_level": "low/medium/high",
    }


def supported_template_ids() -> List[str]:
    return [item["id"] for item in TEMPLATES]

