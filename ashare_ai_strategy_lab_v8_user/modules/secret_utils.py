from __future__ import annotations

import os
from typing import Any


def get_secret(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is not None and str(value).strip() != "":
        return str(value).strip()

    try:
        import streamlit as st  # type: ignore

        if key in st.secrets:
            secret_value: Any = st.secrets[key]
            if secret_value is not None and str(secret_value).strip() != "":
                return str(secret_value).strip()
    except Exception:
        pass

    return default
