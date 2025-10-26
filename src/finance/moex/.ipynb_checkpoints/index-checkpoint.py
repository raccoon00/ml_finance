from __future__ import annotations

r"""
Асинхронное получение списка индексов MOEX (движок stock, рынок index) через aiomoex + aiohttp,
с постраничной загрузкой по параметру start, нормализацией и группировкой по категориям
(общий рынок, секторальные, облигации и др.), а также получение истории выбранного индекса
с расчётом лог-доходностей и выводом компактной сводки при verbose=True.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import aiomoex
import numpy as np
import pandas as pd

from finance.moex.moex_common import PrettyPrinter, compute_log_returns


# ------------------------------- Константы ---------------------------------

ISS_URL = "https://iss.moex.com/iss/engines/stock/markets/index/securities.json"

# Эвристики для маппинга индексов в группы (можно расширять)
CATEGORY_PATTERNS: List[Tuple[str, List[re.Pattern]]] = [
    ("main", [
        re.compile(r"^(IMOEX|RTSI)$"),
        re.compile(r"BMI", re.IGNORECASE),
    ]),
    ("bonds", [
        re.compile(r"^RGBI"),
        re.compile(r"^RUGBI"),
        re.compile(r"bond", re.IGNORECASE),
    ]),
    ("sectoral", [
        re.compile(r"^(MOEX|RTS)[A-Z]{2,}$"),
        re.compile(r"sector", re.IGNORECASE),
    ]),
    ("blue_chip", [
        re.compile(r"(MOEX)?10"),
        re.compile(r"blue", re.IGNORECASE),
        re.compile(r"SMID", re.IGNORECASE),
    ]),
    ("dividend", [
        re.compile(r"TR$"),
        re.compile(r"div", re.IGNORECASE),
        re.compile(r"factor", re.IGNORECASE),
    ]),
]

DEFAULT_CATEGORY = "other"


# --------------------------- Вспомогательные ИСС ---------------------------

async def _fetch_indices(session: aiohttp.ClientSession) -> pd.DataFrame:
    """
    Загрузка индексов с moex. 
    """

    client = aiomoex.ISSClient(session, ISS_URL)
    data = await client.get()
    securities = data["securities"]
    data = pd.DataFrame(securities)
    
    return data


def _normalize_indices_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит список индексов к основным полям и сортирует по SECID. 
    """
    keep = [c for c in ("SECID", "SHORTNAME", "NAME", "ISIN", "LISTLEVEL") if c in df.columns]
    out = df[keep].copy() if keep else df.copy()
    if "SECID" in out.columns:
        out = out.sort_values("SECID").reset_index(drop=True)
    return out


def _detect_category(secid: str, shortname: Optional[str], name: Optional[str]) -> str:
    """
    Назначение категории по шаблонам SECID и эвристикам по названию. 
    """
    s = secid or ""
    sn = shortname or ""
    nm = name or ""
    for cat, pats in CATEGORY_PATTERNS:
        for p in pats:
            if p.search(s) or p.search(sn) or p.search(nm):
                return cat
    return DEFAULT_CATEGORY


def _group_indices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Группировка индексов по категориям, возвращает словарь {категория: DataFrame}. 
    """
    if df.empty:
        return {}
    secid = df["SECID"].astype(str) if "SECID" in df.columns else pd.Series([""], index=df.index)
    shortname = df["SHORTNAME"].astype(str) if "SHORTNAME" in df.columns else pd.Series([""], index=df.index)
    name = df["NAME"].astype(str) if "NAME" in df.columns else pd.Series([""], index=df.index)

    cats: List[str] = []
    for s, sn, nm in zip(secid, shortname, name):
        cats.append(_detect_category(s, sn, nm))

    work = df.copy()
    work["CATEGORY"] = cats
    groups: Dict[str, pd.DataFrame] = {}
    for cat, g in work.groupby("CATEGORY", sort=True):
        groups[cat] = g.drop(columns=["CATEGORY"]).reset_index(drop=True)
    return groups


def _build_list_summary_rows(groups: Dict[str, pd.DataFrame], fetch_seconds: float) -> List[Tuple[str, str, str]]:
    """
    Строки для сводной карточки списка индексов: время fetch, всего индексов и распределение по категориям. 
    """
    total = sum(len(df) for df in groups.values())
    rows: List[Tuple[str, str, str]] = [
        ("Источник", "ISS MOEX (рынок index)", "magenta"),
        ("Время fetch", f"{fetch_seconds*1000:.0f} мс", "cyan"),
        ("Всего индексов", f"{total}", "blue"),
    ]
    order = ["Общий рынок", "Секторальные", "Облигации", "Голубые фишки/укрупнённые", "Дивидендные/факторные", "Прочие"]
    for cat in order:
        if cat in groups:
            rows.append((cat, str(len(groups[cat])), "green"))
    for cat in sorted(set(groups.keys()) - set(order)):
        rows.append((cat, str(len(groups[cat])), "green"))
    return rows


# --------------------------- История по индексу ----------------------------

async def _fetch_index_history(
    session: aiohttp.ClientSession,
    index: str,
    *,
    board: str,
) -> tuple[pd.DataFrame, int]:
    """
    Загрузка дневной истории значений индекса с указанной доски ISS MOEX (engine='stock', market='index'). 
    """
    data: List[Dict[str, Any]] = await aiomoex.get_board_history(
        session,
        index,
        board=board,
        market="index",
        engine="stock",
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df, 0

    # Нормализация индекса дат
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    na_count = df["TRADEDATE"].isna().sum()    
    df = df.dropna(subset=["TRADEDATE"]).set_index("TRADEDATE").sort_index()
        
    return df, na_count


def _build_index_summary_rows(
    index_name: str,
    df: pd.DataFrame,
    fetch_seconds: float,
    na_count: int,
) -> List[Tuple[str, str, str]]:
    """
    Сводка по индексу: диапазон истории, первая/последняя точки, совокупная и годовая доходность. 
    """
    first_dt: Optional[pd.Timestamp] = df.index.min()
    last_dt: Optional[pd.Timestamp] = df.index.max()
    n_days = int(df.shape[0])

    first_val = np.nan
    last_val = np.nan
    if "CLOSE" in df.columns and not df.empty:
        first_val = float(df["CLOSE"].iloc[0]) if pd.notna(df["CLOSE"].iloc[0]) else np.nan
        last_val = float(df["CLOSE"].iloc[-1]) if pd.notna(df["CLOSE"].iloc[-1]) else np.nan

    total_return = np.nan
    if np.isfinite(first_val) and np.isfinite(last_val) and first_val != 0.0:
        total_return = ((last_val / first_val) - 1.0) * 100.0

    annualized_return = np.nan
    if "logret" in df.columns and not df["logret"].isna().all():
        mean_logret = float(df["logret"].mean())
        if np.isfinite(mean_logret):
            annualized_return = (np.exp(mean_logret * 252) - 1.0) * 100.0

    rows: List[Tuple[str, str, str]] = [
        ("Индекс", index_name, "magenta"),
        ("Время fetch", f"{fetch_seconds*1000:.0f} мс", "cyan"),
        ("Количество NaN", f"{na_count}", "cyan"),
        ("История", f"{first_dt.date() if pd.notna(first_dt) else '—'} → {last_dt.date() if pd.notna(last_dt) else '—'}", "blue"),
        ("Торговых дней", f"{n_days}", "blue"),
        ("Первое значение", f"{first_val:.2f}" if np.isfinite(first_val) else "—", "green"),
        ("Последнее значение", f"{last_val:.2f}" if np.isfinite(last_val) else "—", "green"),
        ("Совокупная доходность", f"{total_return:.2f}%" if np.isfinite(total_return) else "—", "yellow"),
        ("Годовая доходность (CAGR)", f"{annualized_return:.2f}%" if np.isfinite(annualized_return) else "—", "yellow"),
    ]
    return rows


# -------------------------------- Публичное API ----------------------------

async def list_indices(verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Загрузка полного списка индексов MOEX (движок stock, рынок index) через aiomoex + aiohttp и разбиение по категориям; 
    возвращает словарь {категория: DataFrame} с полями SECID, SHORTNAME, NAME, ISIN, LISTLEVEL (если доступны). 
    """
    printer = PrettyPrinter(enabled=verbose)
    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        df, na_count = await _fetch_indices(session)
        t1 = time.perf_counter()
        fetch_seconds = t1 - t0

    df = _normalize_indices_df(df)
    groups = _group_indices(df)

    if verbose:
        rows = _build_list_summary_rows(groups, fetch_seconds, na_count)
        printer.summary_card("MOEX индексы (группы)", rows)

    return groups


async def fetch_stock_tickers(index) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        url: str = "https://iss.moex.com" f"/iss/statistics/engines/stock/markets/index/analytics/{index}/tickers.json"
        iss = aiomoex.ISSClient(session, url)
        tickers = await iss.get()
        print(tickers.keys())
    return pd.DataFrame(tickers["tickers"])


async def fetch_index(index: str = "RGBITR", board: str = "SNDX", verbose: bool = False) -> pd.DataFrame:
    """
    Загрузка дневной истории индекса (по умолчанию RGBITR) с указанной доски и расчёт лог-доходностей; 
    возвращает DataFrame со столбцами CLOSE и logret. 
    """
    printer = PrettyPrinter(enabled=verbose)
    index_code = index

    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        df, na_count = await _fetch_index_history(
            session,
            index_code,
            board=board,
        )
        t1 = time.perf_counter()
        fetch_seconds = t1 - t0

        if df.empty:
            if verbose:
                printer.summary_card(
                    f"MOEX {index_code}",
                    [("Время fetch", f"{fetch_seconds*1000:.0f} мс", "cyan"),
                     ("История", "нет данных", "yellow")],
                )
            return pd.DataFrame(columns=["CLOSE", "logret"])

        # Приведение к единообразию: оставить CLOSE и посчитать лог-доходности
        if "CLOSE" in df.columns:
            df = df[["CLOSE"]].copy()
            df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
        else:
            # Возможные альтернативы на некоторых досках (подстраховка)
            value_col = None
            for c in ("INDEXVALUE", "VALUE"):
                if c in df.columns:
                    value_col = c
                    break
            if value_col is None:
                df["CLOSE"] = np.nan
                df = df[["CLOSE"]]
            else:
                df["CLOSE"] = pd.to_numeric(df[value_col], errors="coerce")
                df = df[["CLOSE"]]

        df["logret"] = compute_log_returns(df["CLOSE"])

        if verbose:
            rows = _build_index_summary_rows(index_code, df, fetch_seconds, na_count)
            printer.summary_card(f"MOEX {index_code}", rows)

        return df


# Примеры:
# import asyncio
# groups = asyncio.run(list_indices(verbose=True))
# df_rgbtr = asyncio.run(fetch_index("RGBITR", board="RTSI", verbose=True))
