from __future__ import annotations

import importlib.util
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from modules.secret_utils import get_secret


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BUNDLED_HS300_PATH = os.path.join(DATA_DIR, "hs300_constituents_20241118.csv")


@dataclass
class MarketDataConfig:
    fallback_seed: int = 42
    market_news_ttl_seconds: int = 900


class DataProvider:
    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()
        self._hs300_cache: Optional[pd.DataFrame] = None
        self._hs300_source: str = "未初始化"
        self._all_a_cache: Optional[pd.DataFrame] = None
        self._all_a_source: str = "未初始化"
        self._stock_profile_cache: Dict[str, Dict[str, Any]] = {}
        self._market_news_cache: Dict[str, Dict[str, Any]] = {}
        self._market_news_cache_ts: Dict[str, float] = {}
        self._news_status: Dict[str, str] = {}

    @property
    def hs300_source(self) -> str:
        return self._hs300_source

    @property
    def all_a_source(self) -> str:
        return self._all_a_source

    @property
    def news_status(self) -> Dict[str, str]:
        return dict(self._news_status)

    def akshare_available(self) -> bool:
        return importlib.util.find_spec("akshare") is not None

    def get_stock_daily(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        stock_code = normalize_stock_code(stock_code)
        if self.akshare_available():
            try:
                import akshare as ak  # type: ignore

                raw = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust="qfq",
                )
                rename_map = {
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                }
                raw = raw.rename(columns=rename_map)
                needed = ["date", "open", "high", "low", "close", "volume"]
                if not set(needed).issubset(raw.columns):
                    raise ValueError("daily price columns missing")
                df = raw[needed].copy()
                df["date"] = pd.to_datetime(df["date"])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna().sort_values("date").reset_index(drop=True)
                if not df.empty:
                    return df
            except Exception:
                pass
        return self._mock_stock_daily(stock_code, start_date, end_date)

    def get_hs300_candidates(self) -> pd.DataFrame:
        if self._hs300_cache is not None and not self._hs300_cache.empty:
            return self._hs300_cache.copy()

        fetchers = [
            self._fetch_hs300_weight_csindex,
            self._fetch_hs300_csindex,
            self._fetch_hs300_sina,
            self._fetch_hs300_generic,
            self._fetch_bundled_hs300,
        ]

        for fetcher in fetchers:
            try:
                raw_df, source = fetcher()
                df = self._normalize_code_name_df(raw_df)
                if len(df) >= 250 and df["code"].nunique() >= 250 and not (len(df) == 1 and df.iloc[0]["code"] == "000300"):
                    self._hs300_cache = df
                    self._hs300_source = source
                    return df.copy()
            except Exception:
                continue

        self._hs300_source = "内置极简演示列表"
        self._hs300_cache = pd.DataFrame(
            [
                {"code": "600519", "name": "贵州茅台"},
                {"code": "601318", "name": "中国平安"},
                {"code": "600036", "name": "招商银行"},
                {"code": "300750", "name": "宁德时代"},
                {"code": "000333", "name": "美的集团"},
                {"code": "002594", "name": "比亚迪"},
                {"code": "000001", "name": "平安银行"},
                {"code": "600276", "name": "恒瑞医药"},
            ]
        )
        return self._hs300_cache.copy()

    def get_all_a_stocks(self) -> pd.DataFrame:
        if self._all_a_cache is not None and not self._all_a_cache.empty:
            return self._all_a_cache.copy()

        if self.akshare_available():
            try:
                import akshare as ak  # type: ignore
                df = ak.stock_info_a_code_name()
                df = self._normalize_code_name_df(df)
                if not df.empty:
                    self._all_a_cache = df
                    self._all_a_source = "AKShare / stock_info_a_code_name()"
                    return df.copy()
            except Exception:
                pass

        hs300_df = self.get_hs300_candidates()
        self._all_a_cache = hs300_df.copy()
        self._all_a_source = f"沿用 {self._hs300_source}"
        return self._all_a_cache.copy()

    def get_stock_profile(self, code: str) -> Dict[str, Any]:
        code = normalize_stock_code(code)
        if code in self._stock_profile_cache:
            return dict(self._stock_profile_cache[code])

        validation = self.validate_stock_code(code)
        name = validation.get("name")
        industry_cn = None
        industry_source = "未获取"

        if self.akshare_available() and re.fullmatch(r"\d{6}", code):
            try:
                import akshare as ak  # type: ignore
                info_df = ak.stock_individual_info_em(symbol=code)
                industry_cn = self._extract_industry_from_individual_info(info_df)
                if industry_cn:
                    industry_source = "AKShare / stock_individual_info_em()"
            except Exception:
                pass

        if not industry_cn:
            industry_cn = self._guess_industry_from_name(name or "")
            if industry_cn:
                industry_source = "名称关键词推断"

        industry_en = self._map_industry_cn_to_marketaux(industry_cn)
        industry_query = self._build_news_query(name=name, industry_cn=industry_cn)
        profile = {
            "code": code,
            "name": name,
            "industry_cn": industry_cn,
            "industry_en": industry_en,
            "industry_source": industry_source,
            "industry_query": industry_query,
            "in_hs300": bool(validation.get("in_hs300")),
            "validation_message": validation.get("message"),
        }
        self._stock_profile_cache[code] = dict(profile)
        return profile

    def validate_stock_code(self, code: str) -> Dict[str, Any]:
        code = normalize_stock_code(code)
        hs300_df = self.get_hs300_candidates()
        hit = hs300_df[hs300_df["code"] == code]
        if not hit.empty:
            row = hit.iloc[0]
            return {
                "ok": True,
                "code": code,
                "name": row["name"],
                "in_hs300": True,
                "message": f"已识别：{code} | {row['name']}",
            }

        all_a_df = self.get_all_a_stocks()
        hit_all = all_a_df[all_a_df["code"] == code]
        if not hit_all.empty:
            row = hit_all.iloc[0]
            return {
                "ok": False,
                "code": code,
                "name": row["name"],
                "in_hs300": False,
                "message": f"{code} | {row['name']} 存在，但当前不在沪深300范围内。",
            }

        if code == "000300":
            return {
                "ok": False,
                "code": code,
                "name": "沪深300指数",
                "in_hs300": False,
                "message": "000300 是沪深300指数代码，不是成分股代码。请输入沪深300成分股代码，例如 600519、300750。",
            }

        if re.fullmatch(r"\d{6}", code):
            return {
                "ok": False,
                "code": code,
                "name": None,
                "in_hs300": False,
                "message": f"未识别到 {code}，请检查代码是否正确，或确认其是否属于沪深300成分股。",
            }
        return {
            "ok": False,
            "code": code,
            "name": None,
            "in_hs300": False,
            "message": "请输入 6 位 A 股成分股代码，例如 600519、000001、300750。",
        }

    def get_market_news_bundle(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        stock_industry_cn: Optional[str] = None,
        stock_industry_en: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        profile = self.get_stock_profile(stock_code)
        stock_name = stock_name or profile.get("name") or stock_code
        stock_industry_cn = stock_industry_cn or profile.get("industry_cn")
        stock_industry_en = stock_industry_en or profile.get("industry_en")
        cache_key = f"{stock_code}|{stock_name}|{stock_industry_cn}|{stock_industry_en}"
        now = time.time()
        if (
            not force_refresh
            and cache_key in self._market_news_cache
            and now - self._market_news_cache_ts.get(cache_key, 0) < self.config.market_news_ttl_seconds
        ):
            return dict(self._market_news_cache[cache_key])

        self._news_status = {}
        bundle = self._build_demo_news_bundle(stock_name, stock_industry_cn)
        live = self._fetch_live_market_news(stock_name, stock_industry_cn, stock_industry_en)
        if live:
            bundle.update(live)
            bundle["is_live"] = True
        else:
            bundle["is_live"] = False

        bundle["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        bundle["stock_name"] = stock_name
        bundle["stock_code"] = stock_code
        bundle["stock_industry_cn"] = stock_industry_cn
        bundle["stock_industry_en"] = stock_industry_en
        bundle["status"] = dict(self._news_status)
        self._market_news_cache[cache_key] = dict(bundle)
        self._market_news_cache_ts[cache_key] = now
        return bundle

    def get_demo_news(self, stock_name: str = "目标股票", stock_industry_cn: Optional[str] = None) -> Dict[str, str]:
        industry_text = stock_industry_cn or "大盘蓝筹"
        return {
            "domestic": f"A股盘面围绕 {industry_text} 与红利风格轮动，{stock_name} 所在板块受到资金关注；政策与产业消息仍是短线定价核心。",
            "global": "海外方面，利率预期、科技巨头财报与地缘事件继续影响全球风险偏好，成长与防御资产表现分化。",
        }

    def _build_demo_news_bundle(self, stock_name: str, stock_industry_cn: Optional[str]) -> Dict[str, Any]:
        demo = self.get_demo_news(stock_name, stock_industry_cn)
        return {
            "domestic_articles": [
                {"title": f"{stock_industry_cn or 'A股核心资产'} 板块活跃，市场关注政策与资金轮动", "source": "本地快照", "published_at": "-", "url": "", "summary": demo["domestic"]},
                {"title": f"{stock_name} 所在行业受到短线资金关注", "source": "本地快照", "published_at": "-", "url": "", "summary": demo["domestic"]},
            ],
            "global_articles": [
                {"title": "海外利率预期与科技财报影响全球风险偏好", "source": "本地快照", "published_at": "-", "url": "", "summary": demo["global"]},
                {"title": "全球宏观与地缘事件继续扰动权益市场", "source": "本地快照", "published_at": "-", "url": "", "summary": demo["global"]},
            ],
            "domestic_titles": self._split_demo_to_titles(demo["domestic"]),
            "global_titles": self._split_demo_to_titles(demo["global"]),
            "domestic_text": demo["domestic"],
            "global_text": demo["global"],
            "source_note": "内置演示快照（自动回退）",
            "is_live": False,
        }

    def _fetch_live_market_news(
        self,
        stock_name: str,
        stock_industry_cn: Optional[str],
        stock_industry_en: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        domestic_articles: List[Dict[str, Any]] = []
        global_articles: List[Dict[str, Any]] = []
        source_parts: List[str] = []

        ak_domestic = self._fetch_domestic_news_akshare(stock_name, stock_industry_cn)
        if ak_domestic:
            domestic_articles.extend(ak_domestic)
            source_parts.append("AKShare 财联社电报")
            self._news_status["domestic_akshare"] = f"成功（{len(ak_domestic)}条）"
        else:
            self._news_status["domestic_akshare"] = "未取到"

        marketaux_key = get_secret("MARKETAUX_API_KEY", "")
        if marketaux_key:
            ma_domestic = self._fetch_marketaux_news(
                api_key=marketaux_key,
                countries="cn",
                industries=stock_industry_en,
                language="zh",
                search=self._build_news_query(stock_name, stock_industry_cn),
                limit=5,
            )
            ma_global = self._fetch_marketaux_news(
                api_key=marketaux_key,
                countries="us,gb,jp,eu",
                industries=stock_industry_en,
                language="en",
                search=self._build_news_query(stock_name, stock_industry_cn),
                limit=5,
            )
            if ma_domestic:
                domestic_articles.extend(ma_domestic)
                source_parts.append("Marketaux CN")
                self._news_status["marketaux_cn"] = f"成功（{len(ma_domestic)}条）"
            else:
                self._news_status["marketaux_cn"] = "未取到"
            if ma_global:
                global_articles.extend(ma_global)
                source_parts.append("Marketaux Global")
                self._news_status["marketaux_global"] = f"成功（{len(ma_global)}条）"
            else:
                self._news_status["marketaux_global"] = "未取到"
        else:
            self._news_status["marketaux"] = "未配置 API Key"

        thenews_key = get_secret("THENEWSAPI_API_KEY", "")
        if thenews_key:
            tn_domestic = self._fetch_thenewsapi_news(
                api_key=thenews_key,
                locale="cn",
                language="zh",
                categories="business",
                search=self._build_news_query(stock_name, stock_industry_cn),
                limit=5,
            )
            tn_global = self._fetch_thenewsapi_news(
                api_key=thenews_key,
                locale="us,gb",
                language="en",
                categories="business,tech",
                search=self._build_news_query(stock_name, stock_industry_cn),
                limit=5,
            )
            if tn_domestic:
                domestic_articles.extend(tn_domestic)
                source_parts.append("The News API CN")
                self._news_status["thenewsapi_cn"] = f"成功（{len(tn_domestic)}条）"
            else:
                self._news_status["thenewsapi_cn"] = "未取到"
            if tn_global:
                global_articles.extend(tn_global)
                source_parts.append("The News API Global")
                self._news_status["thenewsapi_global"] = f"成功（{len(tn_global)}条）"
            else:
                self._news_status["thenewsapi_global"] = "未取到"
        else:
            self._news_status["thenewsapi"] = "未配置 API Key"

        domestic_articles = self._dedupe_articles(domestic_articles, max_items=8)
        global_articles = self._dedupe_articles(global_articles, max_items=8)
        if not domestic_articles and not global_articles:
            return None

        domestic_text = "；".join([a["title"] for a in domestic_articles[:5]])
        global_text = "；".join([a["title"] for a in global_articles[:5]])
        return {
            "domestic_articles": domestic_articles,
            "global_articles": global_articles,
            "domestic_titles": [a["title"] for a in domestic_articles],
            "global_titles": [a["title"] for a in global_articles],
            "domestic_text": domestic_text,
            "global_text": global_text,
            "source_note": " + ".join(source_parts) if source_parts else "实时新闻源",
        }

    def _fetch_domestic_news_akshare(self, stock_name: str, stock_industry_cn: Optional[str]) -> List[Dict[str, Any]]:
        if not self.akshare_available():
            return []
        try:
            import akshare as ak  # type: ignore
            df = ak.stock_info_global_cls(symbol="全部")
            if df is None or df.empty:
                return []
            df = df.rename(columns={c: str(c) for c in df.columns})
            title_col = next((c for c in df.columns if "标题" in c), None)
            content_col = next((c for c in df.columns if "内容" in c), None)
            date_col = next((c for c in df.columns if "发布" in c or "时间" in c), None)
            if not title_col:
                return []
            keywords = [k for k in [stock_name, stock_industry_cn, "A股", "政策", "行业"] if k]
            out: List[Dict[str, Any]] = []
            for _, row in df.head(30).iterrows():
                title = str(row.get(title_col, "")).strip()
                body = str(row.get(content_col, "")).strip() if content_col else ""
                if not title:
                    continue
                haystack = f"{title} {body}"
                if keywords and not any(k in haystack for k in keywords):
                    continue
                out.append(
                    {
                        "title": title,
                        "source": "财联社",
                        "published_at": str(row.get(date_col, "")) if date_col else "",
                        "url": "",
                        "summary": body[:120],
                    }
                )
                if len(out) >= 6:
                    break
            if out:
                return out

            # 如果严格筛选为空，就退回最近通用宏观快讯
            for _, row in df.head(6).iterrows():
                title = str(row.get(title_col, "")).strip()
                body = str(row.get(content_col, "")).strip() if content_col else ""
                if title:
                    out.append({"title": title, "source": "财联社", "published_at": str(row.get(date_col, "")) if date_col else "", "url": "", "summary": body[:120]})
            return out
        except Exception:
            return []

    def _fetch_marketaux_news(
        self,
        api_key: str,
        countries: Optional[str],
        industries: Optional[str],
        language: Optional[str],
        search: Optional[str],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        try:
            params = {
                "api_token": api_key,
                "limit": min(limit, 5),
                "filter_entities": "true",
            }
            if countries:
                params["countries"] = countries
            if industries:
                params["industries"] = industries
            if language:
                params["language"] = language
            if search:
                params["search"] = search
            resp = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=4)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("data", []) if isinstance(payload, dict) else []
            return [
                {
                    "title": str(item.get("title", "")).strip(),
                    "source": ((item.get("source") or {}).get("name") if isinstance(item.get("source"), dict) else item.get("source")) or "Marketaux",
                    "published_at": str(item.get("published_at", "")),
                    "url": item.get("url", ""),
                    "summary": str(item.get("description", "") or item.get("snippet", "")).strip(),
                }
                for item in articles
                if str(item.get("title", "")).strip()
            ]
        except Exception:
            return []

    def _fetch_thenewsapi_news(
        self,
        api_key: str,
        locale: Optional[str],
        language: Optional[str],
        categories: Optional[str],
        search: Optional[str],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        try:
            params = {
                "api_token": api_key,
                "limit": min(limit, 5),
                "sort": "published_at",
            }
            if locale:
                params["locale"] = locale
            if language:
                params["language"] = language
            if categories:
                params["categories"] = categories
            if search:
                params["search"] = search
            resp = requests.get("https://api.thenewsapi.com/v1/news/all", params=params, timeout=4)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("data", []) if isinstance(payload, dict) else []
            return [
                {
                    "title": str(item.get("title", "")).strip(),
                    "source": item.get("source", "The News API"),
                    "published_at": str(item.get("published_at", "")),
                    "url": item.get("url", ""),
                    "summary": str(item.get("description", "") or item.get("snippet", "")).strip(),
                }
                for item in articles
                if str(item.get("title", "")).strip()
            ]
        except Exception:
            return []

    def _dedupe_articles(self, articles: List[Dict[str, Any]], max_items: int = 8) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for article in articles:
            title = str(article.get("title", "")).strip()
            if not title:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(article)
            if len(out) >= max_items:
                break
        return out

    def _split_demo_to_titles(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r"[。；;]", text) if p.strip()][:4]

    def _normalize_code_name_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "name"])
        columns = list(df.columns)

        def pick_col(priority_tokens: List[str], fallback_idx: int) -> str:
            lower_map = {str(c).lower(): c for c in columns}
            for token in priority_tokens:
                token_lower = token.lower()
                for key, original in lower_map.items():
                    if key == token_lower:
                        return original
                for key, original in lower_map.items():
                    if token_lower in key:
                        return original
            return columns[fallback_idx] if len(columns) > fallback_idx else columns[0]

        code_col = pick_col(["成分券代码", "成分股代码", "样本代码", "证券代码", "股票代码", "cons_code", "code", "代码"], 0)
        name_col = pick_col(["成分券名称", "成分股名称", "样本名称", "证券简称", "股票简称", "name", "简称", "名称"], 1)

        out = df[[code_col, name_col]].copy()
        out.columns = ["code", "name"]
        out["code"] = out["code"].astype(str).map(normalize_stock_code)
        out["name"] = out["name"].astype(str).str.strip()
        out = out[out["code"].str.fullmatch(r"\d{6}", na=False)]
        out = out.drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
        return out

    def _extract_industry_from_individual_info(self, info_df: pd.DataFrame) -> Optional[str]:
        if info_df is None or info_df.empty:
            return None
        cols = [str(c) for c in info_df.columns]
        if not {"item", "value"}.issubset(set(cols)):
            info_df = info_df.rename(columns={info_df.columns[0]: "item", info_df.columns[1]: "value"})
        for key in ["行业", "所属行业", "东财行业", "申万行业", "证监会行业", "行业分类"]:
            hit = info_df[info_df["item"].astype(str).str.contains(key, na=False)]
            if not hit.empty:
                val = str(hit.iloc[0]["value"]).strip()
                if val and val.lower() != "nan":
                    return val
        return None

    def _guess_industry_from_name(self, name: str) -> Optional[str]:
        guesses = {
            "银行": ["银行"],
            "保险": ["保险"],
            "证券": ["证券", "券商"],
            "白酒": ["茅台", "五粮液", "泸州老窖", "酒"],
            "家电": ["美的", "格力", "海尔"],
            "医药生物": ["医药", "药业", "生物", "医疗", "器械"],
            "电池": ["时代", "锂电", "电池"],
            "汽车整车": ["比亚迪", "汽车"],
            "半导体": ["半导体", "芯片", "微", "电子"],
            "通信": ["通信", "运营商", "中际", "联通", "移动"],
            "煤炭": ["煤"],
            "电力": ["电力", "能源"],
            "食品饮料": ["食品", "饮料"],
        }
        for industry, keywords in guesses.items():
            if any(k in name for k in keywords):
                return industry
        return None

    def _map_industry_cn_to_marketaux(self, industry_cn: Optional[str]) -> Optional[str]:
        if not industry_cn:
            return None
        pairs = {
            "银行": "Financial Services",
            "保险": "Financial Services",
            "证券": "Financial Services",
            "白酒": "Consumer Defensive",
            "食品饮料": "Consumer Defensive",
            "家电": "Consumer Cyclical",
            "医药生物": "Healthcare",
            "电池": "Industrials",
            "汽车整车": "Consumer Cyclical",
            "半导体": "Technology",
            "电子": "Technology",
            "计算机": "Technology",
            "通信": "Communication Services",
            "煤炭": "Energy",
            "电力": "Utilities",
            "有色金属": "Basic Materials",
            "房地产开发": "Real Estate",
            "地产": "Real Estate",
        }
        for key, value in pairs.items():
            if key in industry_cn:
                return value
        return None

    def _build_news_query(self, name: Optional[str], industry_cn: Optional[str]) -> str:
        parts = []
        if name:
            parts.append(f'"{name}"')
        if industry_cn:
            parts.append(industry_cn)
        parts.extend(["A股", "中国股市"])
        return " OR ".join(dict.fromkeys([p for p in parts if p]))

    def _fetch_hs300_weight_csindex(self):
        import akshare as ak  # type: ignore
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        return df, "AKShare / index_stock_cons_weight_csindex(000300)"

    def _fetch_hs300_csindex(self):
        import akshare as ak  # type: ignore
        df = ak.index_stock_cons_csindex(symbol="000300")
        return df, "AKShare / index_stock_cons_csindex(000300)"

    def _fetch_hs300_sina(self):
        import akshare as ak  # type: ignore
        df = ak.index_stock_cons_sina(symbol="000300")
        return df, "AKShare / index_stock_cons_sina(000300)"

    def _fetch_hs300_generic(self):
        import akshare as ak  # type: ignore
        df = ak.index_stock_cons(symbol="000300")
        return df, "AKShare / index_stock_cons(000300)"

    def _fetch_bundled_hs300(self):
        if not os.path.exists(BUNDLED_HS300_PATH):
            raise FileNotFoundError(BUNDLED_HS300_PATH)
        df = pd.read_csv(BUNDLED_HS300_PATH, dtype={"code": str, "name": str})
        return df, "本地快照 / hs300_constituents_20241118.csv"

    def _mock_stock_daily(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash((stock_code, self.config.fallback_seed))) % (2**32))
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        rets = rng.normal(0.0003, 0.018, size=n)
        close = 100 * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.004, size=n))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, size=n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, size=n))
        volume = rng.integers(1_000_000, 12_000_000, size=n)
        return pd.DataFrame({
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume,
        })


def normalize_stock_code(code: str) -> str:
    code = str(code).strip().upper()
    for suffix in [".SH", ".SZ", "SH", "SZ"]:
        if code.endswith(suffix):
            code = code[: -len(suffix)]
    code = code.replace(" ", "")
    if code.isdigit():
        return code.zfill(6)
    return code


def dataframe_preview(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    return df.head(limit).copy()
