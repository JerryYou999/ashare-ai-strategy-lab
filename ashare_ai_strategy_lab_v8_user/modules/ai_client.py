from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from modules.secret_utils import get_secret
from modules.templates import supported_template_ids


DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_GLM_MODEL = "glm-4.7-flash"


@dataclass
class LLMConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: int = 18


class AIClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(
            api_key=get_secret("LLM_API_KEY", ""),
            base_url=get_secret("LLM_BASE_URL", DEFAULT_GLM_BASE_URL),
            model=get_secret("LLM_MODEL", DEFAULT_GLM_MODEL),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.api_key and self.config.base_url)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        if not self.enabled:
            return self._fallback_response(system_prompt, user_prompt)

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            return self._fallback_response(system_prompt, user_prompt)

    def generate_strategy_from_text(self, user_text: str, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        system_prompt = (
            "你是交易策略产品的 AI 策略生成器。"
            "请严格输出 JSON，不要输出 markdown。"
            f"可用模板只有：{supported_template_ids()}。"
            "你必须把自然语言策略映射到现有模板之一，并给出 params。"
        )
        user_prompt = (
            f"用户输入：{user_text}\n"
            f"股票代码：{stock_code}\n开始日期：{start_date}\n结束日期：{end_date}\n"
            "请输出 JSON，字段包括 template_id, stock_code, start_date, end_date, initial_capital, position_size, params, rationale, risk_level。"
        )
        content = self.chat(system_prompt, user_prompt)
        parsed = _try_parse_json(content)
        if parsed is not None:
            return parsed
        return self._fallback_strategy(user_text, stock_code, start_date, end_date)

    def summarize_news(self, domestic_text: str, global_text: str, strategy_text: str) -> str:
        system_prompt = (
            "你是市场情报助手。请用简洁中文输出：\n"
            "1. 今日市场摘要\n2. 受影响行业Top3\n3. 对当前策略的影响（利多/利空/中性）\n4. 风险提示"
        )
        user_prompt = (
            f"国内热点：{domestic_text}\n\n"
            f"海外热点：{global_text}\n\n"
            f"当前策略：{strategy_text}"
        )
        return self.chat(system_prompt, user_prompt)

    def explain_backtest(self, strategy_text: str, metrics: Dict[str, Any]) -> str:
        system_prompt = (
            "你是量化研究助理，请根据回测指标给出策略复盘。"
            "请输出：表现总结、可能有效原因、主要风险、两个优化建议。"
        )
        user_prompt = f"策略：{strategy_text}\n回测指标：{json.dumps(metrics, ensure_ascii=False)}"
        return self.chat(system_prompt, user_prompt)

    def summarize_research(self, research_text: str) -> str:
        system_prompt = (
            "你是策略研究员。请把论文/研报内容总结为：\n"
            "1. 一句话摘要\n2. 策略类型\n3. 交易频率\n4. 核心信号\n5. 入场条件\n6. 出场条件\n7. 风控方式\n8. 所需数据\n9. 局限性\n10. 适合当前系统的简化版策略"
        )
        return self.chat(system_prompt, research_text[:15000], temperature=0.1)

    def _fallback_strategy(self, user_text: str, stock_code: str, start_date: str, end_date: str) -> Dict[str, Any]:
        text = user_text.lower()
        template_id = "double_ma"
        params: Dict[str, Any] = {"short_window": 10, "long_window": 30}
        rationale = "默认使用双均线趋势模板。"

        if any(k in text for k in ["震荡", "reversion", "mean", "回归", "boll"]):
            template_id = "bollinger_mean_reversion"
            params = {"window": 20, "num_std": 2.0}
            rationale = "识别为震荡/均值回归意图，映射到布林带策略。"
        elif any(k in text for k in ["突破", "breakout", "filter", "trend"]):
            template_id = "alexander_filter"
            params = {"filter_pct": 0.05}
            rationale = "识别为突破/趋势意图，映射到 Filter 策略。"
        elif any(k in text for k in ["动量", "momentum"]):
            template_id = "momentum_short_term"
            params = {"lookback": 20, "entry_threshold": 0.03}
            rationale = "识别为动量意图，映射到短期动量策略。"
        elif any(k in text for k in ["反转", "超跌", "contrarian"]):
            template_id = "contrarian_long_term"
            params = {"lookback": 120, "entry_threshold": -0.15, "exit_threshold": 0.05}
            rationale = "识别为反转意图，映射到长期反转策略。"

        return {
            "template_id": template_id,
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 100000,
            "position_size": 0.95,
            "params": params,
            "rationale": rationale,
            "risk_level": "medium",
        }

    def _fallback_response(self, system_prompt: str, user_prompt: str) -> str:
        if "市场情报助手" in system_prompt:
            return (
                "1. 今日市场摘要：国内政策与海外利率预期共同影响风险偏好。\n"
                "2. 受影响行业Top3：半导体、算力、红利资产。\n"
                "3. 对当前策略的影响：中性偏多，若成长风格继续扩散，趋势型策略更受益。\n"
                "4. 风险提示：宏观预期变化可能加大波动，注意回撤控制。"
            )
        if "量化研究助理" in system_prompt:
            return (
                "表现总结：策略具备一定收益能力，但需重点关注回撤与交易频率。\n"
                "可能有效原因：信号捕捉到了阶段性趋势/均值回归特征。\n"
                "主要风险：样本外失效、参数敏感、单一股票暴露。\n"
                "优化建议：增加训练/验证切分；测试不同参数区间的鲁棒性。"
            )
        if "策略研究员" in system_prompt:
            return (
                "1. 一句话摘要：该文提出一种基于市场异象/事件信息的交易思路。\n"
                "2. 策略类型：动量/事件驱动。\n"
                "3. 交易频率：日频。\n"
                "4. 核心信号：当关键变量达到阈值时触发建仓。\n"
                "5. 入场条件：出现正向信号。\n"
                "6. 出场条件：信号反转或达到持有期。\n"
                "7. 风控方式：止损、仓位控制。\n"
                "8. 所需数据：价格、成交量、事件/新闻数据。\n"
                "9. 局限性：对数据质量与样本外稳定性敏感。\n"
                "10. 简化版策略：可映射为短期动量或新闻事件观察模板。"
            )
        return "LLM 未配置，当前为本地演示模式。"


def _try_parse_json(content: str) -> Optional[Dict[str, Any]]:
    content = content.strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except Exception:
            return None
    return None
