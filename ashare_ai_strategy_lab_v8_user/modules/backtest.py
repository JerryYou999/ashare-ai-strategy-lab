from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.0005
    min_lot: int = 100
    position_size: float = 0.95


class StrategyEngine:
    @staticmethod
    def generate_signals(df: pd.DataFrame, template_id: str, params: Dict[str, Any]) -> pd.Series:
        if template_id == "double_ma":
            return StrategyEngine._double_ma(df, params)
        if template_id == "bollinger_mean_reversion":
            return StrategyEngine._bollinger_mean_reversion(df, params)
        if template_id == "alexander_filter":
            return StrategyEngine._alexander_filter(df, params)
        if template_id == "momentum_short_term":
            return StrategyEngine._momentum(df, params)
        if template_id == "contrarian_long_term":
            return StrategyEngine._contrarian(df, params)
        if template_id == "gap_strategy":
            return StrategyEngine._gap(df, params)
        if template_id == "seasonal_halloween":
            return StrategyEngine._seasonal(df, params)
        raise ValueError(f"Unsupported template for auto backtest: {template_id}")

    @staticmethod
    def _double_ma(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        short_window = int(params.get("short_window", 10))
        long_window = int(params.get("long_window", 30))
        short_ma = df["close"].rolling(short_window).mean()
        long_ma = df["close"].rolling(long_window).mean()
        signal = (short_ma > long_ma).astype(int)
        return signal.fillna(0)

    @staticmethod
    def _bollinger_mean_reversion(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        window = int(params.get("window", 20))
        num_std = float(params.get("num_std", 2.0))
        mid = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std(ddof=0)
        lower = mid - num_std * std
        upper = mid + num_std * std

        signal = pd.Series(0, index=df.index, dtype=int)
        in_position = False
        for i in range(len(df)):
            price = df.iloc[i]["close"]
            if np.isnan(mid.iloc[i]) or np.isnan(lower.iloc[i]) or np.isnan(upper.iloc[i]):
                signal.iloc[i] = 0
                continue
            if not in_position and price < lower.iloc[i]:
                in_position = True
            elif in_position and price >= mid.iloc[i]:
                in_position = False
            signal.iloc[i] = int(in_position)
        return signal

    @staticmethod
    def _alexander_filter(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        filter_pct = float(params.get("filter_pct", 0.05))
        signal = pd.Series(0, index=df.index, dtype=int)
        if df.empty:
            return signal

        min_price = df.iloc[0]["close"]
        max_price = df.iloc[0]["close"]
        state = 0
        for i in range(len(df)):
            price = float(df.iloc[i]["close"])
            min_price = min(min_price, price)
            max_price = max(max_price, price)

            if state != 1 and price >= min_price * (1 + filter_pct):
                state = 1
                min_price = price
                max_price = price
            elif state == 1 and price <= max_price * (1 - filter_pct):
                state = 0
                min_price = price
                max_price = price
            signal.iloc[i] = state
        return signal

    @staticmethod
    def _momentum(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        lookback = int(params.get("lookback", 20))
        entry_threshold = float(params.get("entry_threshold", 0.03))
        ret = df["close"] / df["close"].shift(lookback) - 1
        signal = (ret > entry_threshold).astype(int)
        return signal.fillna(0)

    @staticmethod
    def _contrarian(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        lookback = int(params.get("lookback", 120))
        entry_threshold = float(params.get("entry_threshold", -0.15))
        exit_threshold = float(params.get("exit_threshold", 0.05))
        long_ret = df["close"] / df["close"].shift(lookback) - 1
        signal = pd.Series(0, index=df.index, dtype=int)
        in_position = False
        for i in range(len(df)):
            value = long_ret.iloc[i]
            if np.isnan(value):
                continue
            if not in_position and value <= entry_threshold:
                in_position = True
            elif in_position and value >= exit_threshold:
                in_position = False
            signal.iloc[i] = int(in_position)
        return signal

    @staticmethod
    def _gap(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        gap_threshold = float(params.get("gap_threshold", 0.02))
        mode = str(params.get("mode", "reversal")).lower()
        gap = df["open"] / df["close"].shift(1) - 1
        signal = pd.Series(0, index=df.index, dtype=int)
        if mode == "momentum":
            signal[gap > gap_threshold] = 1
        else:
            signal[gap < -gap_threshold] = 1
        return signal.fillna(0)

    @staticmethod
    def _seasonal(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        hold_months = params.get("hold_months", [11, 12, 1, 2, 3, 4])
        months = pd.to_datetime(df["date"]).dt.month
        signal = months.isin(hold_months).astype(int)
        return pd.Series(signal.values, index=df.index, dtype=int)


class Backtester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, df: pd.DataFrame, signal: pd.Series) -> Dict[str, Any]:
        data = df.copy().reset_index(drop=True)
        data["signal"] = signal.reset_index(drop=True).fillna(0).astype(int)
        data["date"] = pd.to_datetime(data["date"])

        cash = float(self.config.initial_capital)
        shares = 0
        entry_price = None
        entry_date = None
        trades: List[Dict[str, Any]] = []
        portfolio_values: List[float] = []
        cash_history: List[float] = []
        shares_history: List[int] = []

        for i in range(len(data)):
            row = data.iloc[i]
            current_open = float(row["open"])
            current_close = float(row["close"])

            # Execute based on yesterday's signal at today's open
            if i > 0:
                desired_position = int(data.iloc[i - 1]["signal"])

                if desired_position == 1 and shares == 0:
                    investable_cash = cash * self.config.position_size
                    lot_cost = current_open * self.config.min_lot
                    qty_lots = int(investable_cash // lot_cost)
                    buy_qty = qty_lots * self.config.min_lot
                    if buy_qty > 0:
                        trade_amount = buy_qty * current_open
                        commission = trade_amount * self.config.commission_rate
                        total_cost = trade_amount + commission
                        if total_cost <= cash:
                            cash -= total_cost
                            shares = buy_qty
                            entry_price = current_open
                            entry_date = row["date"]

                elif desired_position == 0 and shares > 0:
                    trade_amount = shares * current_open
                    commission = trade_amount * self.config.commission_rate
                    stamp_duty = trade_amount * self.config.stamp_duty_rate
                    cash += trade_amount - commission - stamp_duty
                    pnl = (current_open - float(entry_price or current_open)) * shares - commission - stamp_duty
                    holding_days = int((row["date"] - pd.Timestamp(entry_date)).days) if entry_date is not None else 0
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": row["date"],
                            "entry_price": entry_price,
                            "exit_price": current_open,
                            "shares": shares,
                            "pnl": pnl,
                            "holding_days": holding_days,
                        }
                    )
                    shares = 0
                    entry_price = None
                    entry_date = None

            market_value = shares * current_close
            total_value = cash + market_value
            portfolio_values.append(total_value)
            cash_history.append(cash)
            shares_history.append(shares)

        data["portfolio_value"] = portfolio_values
        data["cash"] = cash_history
        data["shares"] = shares_history
        data["portfolio_return"] = data["portfolio_value"].pct_change().fillna(0.0)
        data["benchmark_value"] = self._buy_and_hold_curve(data)
        data["benchmark_return"] = data["benchmark_value"].pct_change().fillna(0.0)
        data["drawdown"] = data["portfolio_value"] / data["portfolio_value"].cummax() - 1
        data["benchmark_drawdown"] = data["benchmark_value"] / data["benchmark_value"].cummax() - 1

        metrics = self._compute_metrics(data, trades)
        return {"equity": data, "trades": pd.DataFrame(trades), "metrics": metrics}

    def _buy_and_hold_curve(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float)
        initial_capital = float(self.config.initial_capital)
        first_open = float(data.iloc[0]["open"])
        lot_cost = first_open * self.config.min_lot
        qty_lots = int((initial_capital // lot_cost))
        shares = qty_lots * self.config.min_lot
        trade_amount = shares * first_open
        commission = trade_amount * self.config.commission_rate
        cash = initial_capital - trade_amount - commission
        return data["close"] * shares + cash

    def _compute_metrics(self, equity: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        daily_returns = equity["portfolio_return"]
        total_return = equity["portfolio_value"].iloc[-1] / equity["portfolio_value"].iloc[0] - 1
        benchmark_return = equity["benchmark_value"].iloc[-1] / equity["benchmark_value"].iloc[0] - 1

        n_days = max(len(equity), 1)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 1 else 0.0
        annual_vol = daily_returns.std(ddof=0) * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 1e-12 else 0.0
        max_drawdown = equity["drawdown"].min()

        if trades:
            trade_df = pd.DataFrame(trades)
            win_rate = float((trade_df["pnl"] > 0).mean())
            avg_holding_days = float(trade_df["holding_days"].mean())
            trade_count = int(len(trade_df))
        else:
            win_rate = 0.0
            avg_holding_days = 0.0
            trade_count = 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "benchmark_return": float(benchmark_return),
            "trade_count": trade_count,
            "win_rate": float(win_rate),
            "avg_holding_days": float(avg_holding_days),
        }


def format_metrics(metrics: Dict[str, float]) -> List[Tuple[str, str]]:
    return [
        ("累计收益", f"{metrics['total_return']:.2%}"),
        ("年化收益", f"{metrics['annual_return']:.2%}"),
        ("年化波动", f"{metrics['annual_volatility']:.2%}"),
        ("Sharpe", f"{metrics['sharpe']:.2f}"),
        ("最大回撤", f"{metrics['max_drawdown']:.2%}"),
        ("基准收益", f"{metrics['benchmark_return']:.2%}"),
        ("交易次数", str(metrics['trade_count'])),
        ("胜率", f"{metrics['win_rate']:.2%}"),
        ("平均持有天数", f"{metrics['avg_holding_days']:.1f}"),
    ]

