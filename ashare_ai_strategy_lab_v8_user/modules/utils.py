from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict


def pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def today_str() -> str:
    return date.today().isoformat()

