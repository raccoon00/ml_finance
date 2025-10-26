from __future__ import annotations

"""
Общая логика для работы с ISS MOEX: адаптивный принтер и вспомогательные утилиты.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


class PrettyPrinter:
    """
    Адаптивный принтер: ANSI-цветной вывод в терминале и HTML-карточки в Jupyter.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._in_jupyter = self._detect_jupyter()
        self._html_ready = False
        if self._in_jupyter:
            try:
                from IPython.display import display, HTML  # type: ignore
                self._display = display
                self._HTML = HTML
                self._html_ready = True
            except Exception:
                self._html_ready = False

    @staticmethod
    def _detect_jupyter() -> bool:
        try:
            from IPython import get_ipython  # type: ignore
            shell = get_ipython().__class__.__name__  # type: ignore
            return shell in ("ZMQInteractiveShell", "Shell")
        except Exception:
            return False

    @staticmethod
    def _ansi(col: str) -> str:
        palette = {
            "blue": "\033[34m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "reset": "\033[0m",
            "bold": "\033[1m",
            "dim": "\033[2m",
        }
        return palette.get(col, "")

    def summary_card(
        self,
        title: str,
        rows: List[Tuple[str, str, str]],  # (label, value, color_key)
    ) -> None:
        """
        Вывод компактной сводки: одна плашка со всей информацией.
        
        Параметры:
        - title: заголовок карточки.
        - rows: список кортежей (метка, значение, цвет).
        """
        if not self.enabled:
            return
        if self._in_jupyter and self._html_ready:
            color_map = {
                "blue": "#1f6feb",
                "green": "#2ea043",
                "yellow": "#d29922",
                "magenta": "#8957e5",
                "cyan": "#39c5cf",
                "muted": "#8b949e",
                "text": "#c9d1d9",
            }
            items = "\n".join(
                f"""
                <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(240,246,252,0.06);">
                    <span style="color:{color_map['muted']};">{label}</span>
                    <span style="font-weight:600; color:{color_map.get(color, color_map['text'])};">{value}</span>
                </div>
                """
                for (label, value, color) in rows
            )
            html = f"""
            <div style="width: 500px; border-left:6px solid #8957e5; padding:12px 14px; margin:10px 0; background:#0d1117; color:#c9d1d9; border-radius:8px;">
                <div style="font-weight:700; font-size:14px; color:#8957e5; margin-bottom:6px;">📌 {title}</div>
                {items}
            </div>
            """
            self._display(self._HTML(html))  # type: ignore
        else:
            # Терминал: компактный блок с ANSI
            bar = f"{self._ansi('magenta')}{'='*6}{self._ansi('reset')}"
            print(f"{bar} {self._ansi('magenta')}{title}{self._ansi('reset')} {bar}")
            for label, value, color in rows:
                col = self._ansi(color) if color in ("blue", "green", "yellow", "magenta", "cyan") else ""
                print(f"{self._ansi('dim')}{label:<26}{self._ansi('reset')}{col}{value}{self._ansi('reset')}")


def compute_log_returns(series: pd.Series) -> pd.Series:
    """
    Расчёт логарифмических доходностей по ценовому ряду.
    
    Параметры:
    - series: ряд цен (или adj_close).
    
    Возвращает:
    - pd.Series лог-доходностей: ln(series / series.shift(1)).
    """
    return np.log(series / series.shift(1))
