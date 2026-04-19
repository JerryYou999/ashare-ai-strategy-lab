from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SectorMappingResult:
    sectors: List[str]
    sentiment: str


SECTOR_KEYWORDS = {
    "半导体": ["芯片", "半导体", "光刻", "算力", "gpu", "ai服务器"],
    "新能源": ["锂电", "光伏", "储能", "新能源车", "风电"],
    "金融": ["降准", "降息", "银行", "保险", "券商"],
    "消费": ["消费", "白酒", "餐饮", "旅游", "家电"],
    "地产链": ["地产", "基建", "建材", "家居"],
    "高股息": ["红利", "煤炭", "电力", "运营商", "央企"],
}

POSITIVE_WORDS = ["利好", "增长", "修复", "突破", "改善", "上调", "扩张", "创新高"]
NEGATIVE_WORDS = ["利空", "下调", "制裁", "回落", "衰退", "风险", "波动", "收缩"]


class NewsEngine:
    def map_to_sectors(self, domestic_text: str, global_text: str) -> SectorMappingResult:
        text = f"{domestic_text} {global_text}".lower()
        scores: Dict[str, int] = {}
        for sector, keywords in SECTOR_KEYWORDS.items():
            scores[sector] = sum(1 for kw in keywords if kw.lower() in text)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sectors = [name for name, score in ranked if score > 0][:3]
        if not sectors:
            sectors = ["科技成长", "高股息", "顺周期"]

        pos = sum(word in text for word in POSITIVE_WORDS)
        neg = sum(word in text for word in NEGATIVE_WORDS)
        if pos > neg:
            sentiment = "利多"
        elif neg > pos:
            sentiment = "利空"
        else:
            sentiment = "中性"
        return SectorMappingResult(sectors=sectors, sentiment=sentiment)

    def build_strategy_context(self, strategy_name: str, params: dict) -> str:
        return f"策略名称：{strategy_name}；参数：{params}"

