from __future__ import annotations

import copy
from datetime import date, timedelta
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.ai_client import AIClient
from modules.backtest import BacktestConfig, Backtester, StrategyEngine, format_metrics
from modules.data_provider import DataProvider, normalize_stock_code
from modules.news_engine import NewsEngine
from modules.research_engine import ResearchEngine
from modules.templates import get_template, list_templates


st.set_page_config(page_title="A股 AI 策略实验室", layout="wide")

provider = DataProvider()
ai_client = AIClient()
news_engine = NewsEngine()
research_engine = ResearchEngine()


def init_session() -> None:
    if "current_strategy" not in st.session_state:
        tpl = get_template("double_ma")
        st.session_state.current_strategy = {
            "template_id": tpl["id"],
            "template_name": tpl["name"],
            "stock_code": "600519",
            "stock_name": "贵州茅台",
            "start_date": (date.today() - timedelta(days=365 * 3)).isoformat(),
            "end_date": date.today().isoformat(),
            "initial_capital": 100000,
            "position_size": tpl["default_params"].get("position_size", 0.95),
            "params": copy.deepcopy(tpl["default_params"]),
            "rationale": tpl["description"],
            "risk_level": "medium",
            "auto_backtest": tpl["auto_backtest"],
        }
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None
    if "backtest_explanation" not in st.session_state:
        st.session_state.backtest_explanation = ""
    if "research_summary" not in st.session_state:
        st.session_state.research_summary = ""
    if "research_candidates" not in st.session_state:
        st.session_state.research_candidates = []
    if "research_raw_len" not in st.session_state:
        st.session_state.research_raw_len = 0
    if "news_refresh_nonce" not in st.session_state:
        st.session_state.news_refresh_nonce = 0
    if "news_summary" not in st.session_state:
        st.session_state.news_summary = ""
    if "news_mapping" not in st.session_state:
        st.session_state.news_mapping = None
    if "news_bundle_signature" not in st.session_state:
        st.session_state.news_bundle_signature = ""


init_session()


@st.cache_data(ttl=900, show_spinner=False)
def get_market_bundle(refresh_nonce: int, stock_code: str, stock_name: str, stock_industry_cn: str, stock_industry_en: str) -> Dict[str, Any]:
    return provider.get_market_news_bundle(
        stock_code=stock_code,
        stock_name=stock_name,
        stock_industry_cn=stock_industry_cn,
        stock_industry_en=stock_industry_en,
        force_refresh=refresh_nonce > 0,
    )


def build_param_explanations(template_id: str, params: Dict[str, Any], position_size: float) -> List[str]:
    lines: List[str] = []
    if template_id == "double_ma":
        short_w = int(params.get("short_window", 10))
        long_w = int(params.get("long_window", 30))
        lines = [
            f"短均线 = {short_w} 天：越小越灵敏，越容易更早发出买卖信号，但也更容易被噪音干扰。",
            f"长均线 = {long_w} 天：越大越稳健，适合过滤短期波动，但信号会更慢。",
            "两条均线间距越大，通常交易频率越低；间距越小，信号切换更频繁。",
        ]
    elif template_id == "bollinger_mean_reversion":
        lines = [
            f"布林带窗口 = {int(params.get('window', 20))}：越大越平滑，更强调长期均值；越小越敏感。",
            f"标准差倍数 = {float(params.get('num_std', 2.0)):.1f}：越大越不容易入场，交易次数通常更少。",
            "这个模板更适合震荡市；若市场单边趋势很强，均值回归信号容易失效。",
        ]
    elif template_id == "alexander_filter":
        filter_pct = float(params.get("filter_pct", 0.05))
        lines = [
            f"过滤阈值 = {filter_pct:.0%}：阈值越高，需要更明显的突破才入场，交易次数会减少。",
            "较低阈值更容易跟到短趋势，但也更容易产生假突破。",
            "较高阈值通常会减少交易次数，但有可能错过早期趋势。",
        ]
    elif template_id == "momentum_short_term":
        lines = [
            f"观察窗口 = {int(params.get('lookback', 20))}：越短越偏短线，越长越偏中期趋势。",
            f"入场阈值 = {float(params.get('entry_threshold', 0.03)):.0%}：越高越保守，只在更强动量下入场。",
            "如果你发现交易过于频繁，可以优先提高阈值或拉长观察窗口。",
        ]
    elif template_id == "contrarian_long_term":
        lines = [
            f"长期观察窗口 = {int(params.get('lookback', 120))}：越长越强调“长期超跌”。",
            f"超跌阈值 = {float(params.get('entry_threshold', -0.15)):.0%}：设得更负，表示只有跌得更深才考虑入场。",
            f"退出阈值 = {float(params.get('exit_threshold', 0.05)):.0%}：越低越容易提前离场，越高越愿意等待反弹。",
        ]
    elif template_id == "gap_strategy":
        lines = [
            f"跳空阈值 = {float(params.get('gap_threshold', 0.02)):.1%}：越高越只关注更强的开盘异动。",
            f"当前模式 = {params.get('mode', 'reversal')}：reversal 代表博反转，momentum 代表追延续。",
            "这个模板对开盘缺口比较敏感，更适合消息驱动或高波动阶段。",
        ]
    elif template_id == "seasonal_halloween":
        hold_months = params.get("hold_months", [11, 12, 1, 2, 3, 4])
        lines = [
            f"当前持仓月份 = {hold_months}：这些月份内默认持有，其余月份空仓。",
            "增加持仓月份会让策略更接近 Buy & Hold；减少月份会更强调季节过滤。",
            "这个模板更偏教学和研究展示，适合用来验证季节效应思路。",
        ]
    else:
        lines = [
            "当前模板主要用于研究展示，建议优先结合新闻或论文结论做策略假设。",
            "若需要自动回测，建议先转写到双均线、动量或均值回归模板。",
        ]

    lines.append(f"仓位上限 = {position_size:.0%}：越高越激进，收益和回撤都会被放大。")
    return lines


def reset_backtest_cache() -> None:
    st.session_state.backtest_result = None
    st.session_state.backtest_explanation = ""


def stock_display_name(code: str) -> str:
    validation = provider.validate_stock_code(code)
    if validation.get("name"):
        return f"{validation['code']} | {validation['name']}"
    return normalize_stock_code(code)


st.title("A股 AI 策略实验室")
st.caption("面向交易学习者的 AI 策略生成、回测、市场情报与研究助手。仅用于研究与学习，不构成投资建议。")

hs300_df = provider.get_hs300_candidates()
current_validation = provider.validate_stock_code(st.session_state.current_strategy["stock_code"])
current_profile = provider.get_stock_profile(st.session_state.current_strategy["stock_code"])

with st.sidebar:
    st.subheader("系统状态")
    st.write(f"LLM 状态：{'已配置' if ai_client.enabled else '未配置（本地演示模式）'}")
    st.write("市场规则：只做多 / T+1 / 100 股整数手 / 日频回测")
    st.write(f"沪深300名单来源：{provider.hs300_source}")
    st.write(f"当前可校验股票数：{len(hs300_df)}")
    if len(hs300_df) < 250:
        st.warning("当前名单数量异常，系统会继续自动回退到本地完整快照。")
    if provider.hs300_source.startswith("本地快照"):
        st.info("当前已自动回退到本地完整名单快照，因此依然可以完成沪深300范围校验。")
    elif provider.hs300_source.startswith("内置极简"):
        st.warning("当前只加载到极简演示列表，建议稍后刷新或检查部署环境网络。")
    if current_validation.get("ok"):
        st.success(f"当前股票：{current_validation['code']} | {current_validation['name']}")
        st.write(f"所属行业：{current_profile.get('industry_cn') or '未识别'}")
        if current_profile.get("industry_source"):
            st.caption(f"行业来源：{current_profile.get('industry_source')}")


tab1, tab2, tab3, tab4, tab5 = st.tabs(["策略实验室", "模板市场", "回测结果", "市场情报", "策略研究员"])


with tab1:
    st.subheader("策略实验室")
    current = copy.deepcopy(st.session_state.current_strategy)

    left, right = st.columns([1.2, 1.8])
    with left:
        st.markdown("### 股票输入")
        code_input = st.text_input(
            "输入沪深300成分股代码",
            value=current["stock_code"],
            help="这里只接受沪深300成分股代码，不接受指数代码 000300 本身。系统会自动显示对应股票名称。",
        )
        code_input = normalize_stock_code(code_input)
        validation = provider.validate_stock_code(code_input)
        if validation["ok"]:
            st.success(validation["message"])
        else:
            st.error(validation["message"])

        with st.expander("查看示范代码（可直接复制输入）", expanded=False):
            st.dataframe(hs300_df.head(20), use_container_width=True, hide_index=True)
            st.caption(f"当前已加载 {len(hs300_df)} 只可校验股票；示范区仅展示前 20 只。")

    with right:
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("开始日期", value=pd.to_datetime(current["start_date"]).date())
        with c2:
            end_date = st.date_input("结束日期", value=pd.to_datetime(current["end_date"]).date())

        c3, c4, c5 = st.columns(3)
        with c3:
            initial_capital = st.number_input("初始资金", min_value=10000, value=int(current["initial_capital"]), step=10000)
        with c4:
            risk_level = st.selectbox(
                "风险偏好",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(current.get("risk_level", "medium")),
            )
        with c5:
            source = st.radio("策略来源", ["模板/手动", "自然语言生成"], horizontal=False)

    if source == "自然语言生成":
        user_text = st.text_area(
            "用自然语言描述你的策略",
            value="给我一个适合震荡市场的中风险策略，适用于 A 股大盘蓝筹。",
            height=120,
        )
        if st.button("AI 生成策略", key="ai_generate_strategy"):
            if not validation["ok"]:
                st.error("请先输入并确认一个属于沪深300的有效股票代码。")
            else:
                generated = ai_client.generate_strategy_from_text(
                    user_text=user_text,
                    stock_code=validation["code"],
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
                template_meta = get_template(generated["template_id"])
                st.session_state.current_strategy = {
                    **generated,
                    "stock_name": provider.validate_stock_code(generated.get("stock_code", validation["code"])).get("name", validation["name"]),
                    "template_name": template_meta["name"],
                    "auto_backtest": template_meta["auto_backtest"],
                }
                reset_backtest_cache()
                st.success("AI 已生成策略，并写入当前会话。")
                st.rerun()

    st.markdown("### 当前策略")
    current = copy.deepcopy(st.session_state.current_strategy)
    st.write(f"模板：**{current['template_name']}**")
    st.write(f"股票：**{stock_display_name(current['stock_code'])}**")
    st.write(f"策略说明：{current.get('rationale', '-')}")

    with st.expander("编辑当前模板参数", expanded=True):
        tpl_id = current["template_id"]
        params = copy.deepcopy(current["params"])

        if tpl_id == "double_ma":
            params["short_window"] = st.slider("短均线", 3, 60, int(params.get("short_window", 10)))
            params["long_window"] = st.slider("长均线", 10, 180, int(params.get("long_window", 30)))
        elif tpl_id == "bollinger_mean_reversion":
            params["window"] = st.slider("布林带窗口", 5, 80, int(params.get("window", 20)))
            params["num_std"] = st.slider("标准差倍数", 1.0, 3.5, float(params.get("num_std", 2.0)), 0.1)
        elif tpl_id == "alexander_filter":
            params["filter_pct"] = st.slider("过滤阈值", 0.01, 0.20, float(params.get("filter_pct", 0.05)), 0.01)
        elif tpl_id == "momentum_short_term":
            params["lookback"] = st.slider("观察窗口", 5, 120, int(params.get("lookback", 20)))
            params["entry_threshold"] = st.slider("入场阈值", 0.01, 0.20, float(params.get("entry_threshold", 0.03)), 0.01)
        elif tpl_id == "contrarian_long_term":
            params["lookback"] = st.slider("长期观察窗口", 20, 250, int(params.get("lookback", 120)))
            params["entry_threshold"] = st.slider("超跌阈值", -0.50, -0.01, float(params.get("entry_threshold", -0.15)), 0.01)
            params["exit_threshold"] = st.slider("退出阈值", -0.10, 0.20, float(params.get("exit_threshold", 0.05)), 0.01)
        elif tpl_id == "gap_strategy":
            params["gap_threshold"] = st.slider("跳空阈值", 0.005, 0.10, float(params.get("gap_threshold", 0.02)), 0.005)
            params["mode"] = st.selectbox(
                "模式",
                ["reversal", "momentum"],
                index=0 if params.get("mode", "reversal") == "reversal" else 1,
            )
        elif tpl_id == "seasonal_halloween":
            params["hold_months"] = st.multiselect(
                "持仓月份",
                options=list(range(1, 13)),
                default=params.get("hold_months", [11, 12, 1, 2, 3, 4]),
            )
        elif tpl_id == "news_event_beta":
            params["event_threshold"] = st.slider("事件阈值", 0.1, 1.0, float(params.get("event_threshold", 0.6)), 0.05)

        position_size = st.slider("仓位上限", 0.10, 1.00, float(current.get("position_size", 0.95)), 0.05)
        if st.button("保存当前策略", key="save_current_strategy"):
            if not validation["ok"]:
                st.error("请先输入一个属于沪深300的有效股票代码。")
            else:
                st.session_state.current_strategy = {
                    **current,
                    "stock_code": validation["code"],
                    "stock_name": validation["name"],
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "initial_capital": int(initial_capital),
                    "risk_level": risk_level,
                    "position_size": float(position_size),
                    "params": params,
                }
                reset_backtest_cache()
                st.success("当前策略已保存。")
                st.rerun()

    st.markdown("### 参数解释")
    for line in build_param_explanations(current["template_id"], current["params"], current.get("position_size", 0.95)):
        st.markdown(f"- {line}")


with tab2:
    st.subheader("模板市场")
    st.caption("选择一个模板作为起点，再到“策略实验室”调整参数。")

    templates = list_templates()
    cols = st.columns(2)
    for idx, tpl in enumerate(templates):
        with cols[idx % 2]:
            st.markdown(f"### {tpl['name']}")
            st.write(f"类型：{tpl['category']}")
            st.write(tpl["description"])
            st.write(f"自动回测：{'支持' if tpl['auto_backtest'] else 'Beta / 暂不默认支持'}")
            if st.button(f"使用模板：{tpl['name']}", key=f"use_tpl_{tpl['id']}"):
                current = st.session_state.current_strategy
                st.session_state.current_strategy = {
                    "template_id": tpl["id"],
                    "template_name": tpl["name"],
                    "stock_code": current["stock_code"],
                    "stock_name": current.get("stock_name", provider.validate_stock_code(current["stock_code"]).get("name")),
                    "start_date": current["start_date"],
                    "end_date": current["end_date"],
                    "initial_capital": current["initial_capital"],
                    "position_size": tpl["default_params"].get("position_size", 0.95),
                    "params": copy.deepcopy(tpl["default_params"]),
                    "rationale": tpl["description"],
                    "risk_level": current.get("risk_level", "medium"),
                    "auto_backtest": tpl["auto_backtest"],
                }
                reset_backtest_cache()
                st.success(f"已切换到模板：{tpl['name']}")
                st.rerun()
            st.divider()


with tab3:
    st.subheader("回测结果")
    strategy = copy.deepcopy(st.session_state.current_strategy)
    strategy_label = stock_display_name(strategy["stock_code"])
    st.write(f"当前策略：**{strategy['template_name']}** | 股票：**{strategy_label}**")
    st.write(strategy.get("rationale", ""))

    validation = provider.validate_stock_code(strategy["stock_code"])
    if not validation["ok"]:
        st.error("当前股票未通过沪深300校验，请回到“策略实验室”修正后再运行回测。")
    elif not strategy.get("auto_backtest", True):
        st.warning("当前模板为 Beta 研究模板，默认不支持自动回测。你可以先在“策略研究员”页转写为可实验策略。")
    else:
        if st.button("运行回测", key="run_backtest"):
            df = provider.get_stock_daily(strategy["stock_code"], strategy["start_date"], strategy["end_date"])
            signal = StrategyEngine.generate_signals(df, strategy["template_id"], strategy["params"])
            backtester = Backtester(
                BacktestConfig(
                    initial_capital=float(strategy["initial_capital"]),
                    position_size=float(strategy.get("position_size", 0.95)),
                )
            )
            st.session_state.backtest_result = backtester.run(df, signal)
            st.session_state.backtest_explanation = ""

        result = st.session_state.backtest_result
        if result is not None:
            metrics = result["metrics"]
            equity = result["equity"]
            trades = result["trades"]

            metric_cols = st.columns(3)
            formatted = format_metrics(metrics)
            for i, (label, value) in enumerate(formatted):
                with metric_cols[i % 3]:
                    st.metric(label, value)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity["date"], y=equity["portfolio_value"], mode="lines", name="Strategy"))
            fig.add_trace(go.Scatter(x=equity["date"], y=equity["benchmark_value"], mode="lines", name="Buy & Hold"))
            fig.update_layout(title="策略净值 vs Buy & Hold", xaxis_title="Date", yaxis_title="Portfolio Value", height=420)
            st.plotly_chart(fig, use_container_width=True)

            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(x=equity["date"], y=equity["drawdown"], mode="lines", name="Strategy DD"))
            dd_fig.add_trace(go.Scatter(x=equity["date"], y=equity["benchmark_drawdown"], mode="lines", name="Benchmark DD"))
            dd_fig.update_layout(title="回撤曲线", xaxis_title="Date", yaxis_title="Drawdown", height=320)
            st.plotly_chart(dd_fig, use_container_width=True)

            if st.button("AI 解释回测结果", key="ai_explain_backtest"):
                st.session_state.backtest_explanation = ai_client.explain_backtest(
                    strategy_text=strategy_label + "\n" + strategy.get("rationale", ""),
                    metrics=metrics,
                )

            if st.session_state.backtest_explanation:
                st.markdown("### AI 策略复盘")
                st.write(st.session_state.backtest_explanation)

            st.markdown("### 交易明细")
            if isinstance(trades, pd.DataFrame) and not trades.empty:
                st.dataframe(trades, use_container_width=True)
            else:
                st.info("当前样本期内没有形成完整交易。")
        else:
            st.info("请先运行回测。")


with tab4:
    st.subheader("市场情报")
    strategy = copy.deepcopy(st.session_state.current_strategy)
    profile = provider.get_stock_profile(strategy["stock_code"])

    top1, top2 = st.columns([0.18, 0.82])
    with top1:
        if st.button("刷新情报", key="refresh_news"):
            st.session_state.news_refresh_nonce += 1
            st.rerun()

    bundle = get_market_bundle(
        st.session_state.news_refresh_nonce,
        strategy["stock_code"],
        profile.get("name") or strategy.get("stock_name") or strategy["stock_code"],
        profile.get("industry_cn") or "",
        profile.get("industry_en") or "",
    )

    strategy_context = (
        f"策略：{strategy['template_name']}；"
        f"股票：{stock_display_name(strategy['stock_code'])}；"
        f"行业：{profile.get('industry_cn') or '未知'}；"
        f"参数：{strategy['params']}"
    )
    bundle_sig = bundle.get("domestic_text", "") + "||" + bundle.get("global_text", "") + "||" + strategy_context
    if bundle_sig != st.session_state.news_bundle_signature:
        st.session_state.news_bundle_signature = bundle_sig
        st.session_state.news_mapping = news_engine.map_to_sectors(bundle.get("domestic_text", ""), bundle.get("global_text", ""))
        st.session_state.news_summary = ai_client.summarize_news(bundle.get("domestic_text", ""), bundle.get("global_text", ""), strategy_context)

    with top2:
        st.write(f"当前股票：**{stock_display_name(strategy['stock_code'])}**")
        st.write(f"所属行业：**{profile.get('industry_cn') or '未识别'}**")
        st.caption(
            "情报抓取逻辑：优先 AKShare 财联社电报，再叠加 Marketaux 与 The News API；"
            "如果你还没配置新闻 API Key，页面仍会显示 AKShare / 本地快照。"
        )
        if bundle.get("is_live"):
            st.success(f"已获取实时情报 | 更新时间：{bundle['updated_at']} | 来源：{bundle['source_note']}")
        else:
            st.warning(f"实时新闻源当前不可用，已切换到回退内容 | 更新时间：{bundle['updated_at']} | 来源：{bundle['source_note']}")

    with st.expander("查看数据源状态", expanded=False):
        status = bundle.get("status", {})
        if not status:
            st.info("暂无状态信息。")
        else:
            for k, v in status.items():
                st.write(f"- {k}: {v}")
        st.caption("建议在部署环境中配置 MARKETAUX_API_KEY 和 THENEWSAPI_API_KEY，以提升市场情报页稳定性。")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 国内热点")
        domestic_articles = bundle.get("domestic_articles", [])
        if domestic_articles:
            for idx, article in enumerate(domestic_articles[:6], start=1):
                st.markdown(f"**{idx}. {article.get('title', '-') }**")
                meta = f"来源：{article.get('source', '-') }"
                if article.get("published_at"):
                    meta += f" | 时间：{article.get('published_at')}"
                st.caption(meta)
                if article.get("summary"):
                    st.write(article.get("summary"))
        else:
            st.info("当前没有抓取到国内热点。")

    with col_b:
        st.markdown("### 海外热点")
        global_articles = bundle.get("global_articles", [])
        if global_articles:
            for idx, article in enumerate(global_articles[:6], start=1):
                st.markdown(f"**{idx}. {article.get('title', '-') }**")
                meta = f"来源：{article.get('source', '-') }"
                if article.get("published_at"):
                    meta += f" | 时间：{article.get('published_at')}"
                st.caption(meta)
                if article.get("summary"):
                    st.write(article.get("summary"))
        else:
            st.info("当前没有抓取到海外热点。")

    mapping = st.session_state.news_mapping
    if mapping is not None:
        st.markdown("### 行业映射")
        mc1, mc2, mc3 = st.columns([1, 1, 2])
        with mc1:
            st.metric("当前股票行业", profile.get("industry_cn") or "未识别")
        with mc2:
            st.metric("情绪方向", mapping.sentiment)
        with mc3:
            st.write("受影响行业：" + "、".join(mapping.sectors))

    if st.session_state.news_summary:
        st.markdown("### AI 市场情报解读")
        st.write(st.session_state.news_summary)


with tab5:
    st.subheader("策略研究员 / Research Copilot")
    st.caption("上传论文或粘贴摘要，AI 自动提炼策略结构，并转写为当前系统可实验的候选策略。")

    uploaded_file = st.file_uploader("上传 PDF / 文本文件", type=["pdf", "txt", "md"], key="research_uploader")
    manual_research_text = st.text_area(
        "或直接粘贴论文摘要 / 方法段落",
        value="",
        height=180,
        placeholder="例如粘贴一段关于动量、均值回归、事件驱动或因子策略的摘要……",
    )
    mode = st.selectbox("输出深度", ["快速摘要", "策略提炼", "深度研究"], index=1)

    if st.button("解析并生成研究摘要", key="parse_research"):
        text = ""
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".pdf"):
                text = research_engine.extract_text_from_pdf(uploaded_file.read())
            else:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif manual_research_text.strip():
            text = manual_research_text.strip()

        if not text.strip():
            st.error("请先上传文件，或粘贴至少一段论文/研报内容。")
        else:
            summary = ai_client.summarize_research(text[:15000])
            candidates = research_engine.build_candidates(summary)
            st.session_state.research_summary = summary
            st.session_state.research_candidates = candidates
            st.session_state.research_raw_len = len(text)

    if not st.session_state.research_summary and not st.session_state.research_candidates:
        st.info("这里不会再是空白页。你可以直接上传 PDF，或先粘贴一段摘要做演示。")
        st.markdown(
            "- 动量论文：通常会被转写为“短期动量”模板\n"
            "- 均值回归论文：通常会被转写为“布林带均值回归”模板\n"
            "- 事件驱动论文：通常会被转写为“新闻事件漂移（Beta）”模板"
        )

    if st.session_state.research_summary:
        st.markdown("### 研究摘要")
        st.write(f"原始文本长度：{st.session_state.research_raw_len} 字符 | 输出模式：{mode}")
        st.write(st.session_state.research_summary)

    if st.session_state.research_candidates:
        st.markdown("### 候选策略卡片")
        for idx, candidate in enumerate(st.session_state.research_candidates):
            st.markdown(f"#### {candidate.title}")
            st.write(candidate.rationale)
            st.write(f"模板：{candidate.template_id} | 自动回测：{'支持' if candidate.auto_backtest else '暂不默认支持'}")
            if st.button(f"导入策略实验室：{candidate.title}", key=f"import_candidate_{idx}"):
                tpl = get_template(candidate.template_id)
                current = st.session_state.current_strategy
                validation = provider.validate_stock_code(current["stock_code"])
                st.session_state.current_strategy = {
                    "template_id": candidate.template_id,
                    "template_name": tpl["name"],
                    "stock_code": current["stock_code"],
                    "stock_name": validation.get("name", current.get("stock_name")),
                    "start_date": current["start_date"],
                    "end_date": current["end_date"],
                    "initial_capital": current["initial_capital"],
                    "position_size": candidate.params.get("position_size", tpl["default_params"].get("position_size", 0.95)),
                    "params": candidate.params,
                    "rationale": candidate.rationale,
                    "risk_level": current.get("risk_level", "medium"),
                    "auto_backtest": candidate.auto_backtest,
                }
                reset_backtest_cache()
                st.success("候选策略已导入到策略实验室。")
                st.rerun()
            st.divider()
