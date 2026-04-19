from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List

from pypdf import PdfReader


@dataclass
class ResearchCandidate:
    title: str
    template_id: str
    rationale: str
    params: Dict
    auto_backtest: bool = True


class ResearchEngine:
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts: List[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            texts.append(text)
        return "\n".join(texts)

    def build_candidates(self, summary_text: str) -> List[ResearchCandidate]:
        text = summary_text.lower()
        candidates: List[ResearchCandidate] = []

        if any(k in text for k in ["momentum", "动量"]):
            candidates.append(
                ResearchCandidate(
                    title="论文策略映射：短期动量",
                    template_id="momentum_short_term",
                    rationale="从文档中识别出趋势/动量特征，映射为当前系统的短期动量模板。",
                    params={"lookback": 20, "entry_threshold": 0.03},
                )
            )
        if any(k in text for k in ["reversion", "mean", "均值回归", "bollinger"]):
            candidates.append(
                ResearchCandidate(
                    title="论文策略映射：布林带均值回归",
                    template_id="bollinger_mean_reversion",
                    rationale="文档强调价格偏离后的回归特征，适合先映射为布林带策略验证。",
                    params={"window": 20, "num_std": 2.0},
                )
            )
        if any(k in text for k in ["event", "news", "earnings", "公告", "事件", "新闻"]):
            candidates.append(
                ResearchCandidate(
                    title="论文策略映射：新闻事件漂移（Beta）",
                    template_id="news_event_beta",
                    rationale="文档与事件/新闻/公告相关，建议先作为事件驱动研究模板。",
                    params={"event_threshold": 0.6},
                    auto_backtest=False,
                )
            )
        if any(k in text for k in ["breakout", "filter", "突破"]):
            candidates.append(
                ResearchCandidate(
                    title="论文策略映射：Filter 突破",
                    template_id="alexander_filter",
                    rationale="文档含突破特征，可映射为 Filter 突破模板。",
                    params={"filter_pct": 0.05},
                )
            )

        if not candidates:
            candidates.append(
                ResearchCandidate(
                    title="默认候选：双均线趋势",
                    template_id="double_ma",
                    rationale="未识别到明确策略类型，先映射到最通用的双均线趋势模板。",
                    params={"short_window": 10, "long_window": 30},
                )
            )
        return candidates

