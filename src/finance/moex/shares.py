from __future__ import annotations

r"""
Модуль для асинхронной загрузки дневной истории торгов, дивидендов и сплитов/объединений по бумаге с ISS MOEX
и вычисления скорректированной цены закрытия (adj_close) и логарифмических доходностей (logret).

Функциональность:
- Загрузка свечей по доске торгов (open, high, low, close, volume) через aiomoex.
- Загрузка денежных дивидендов и выравнивание по дате закрытия реестра.
- Загрузка корпоративных действий (сплиты/объединения) и расчёт коэффициента разбиения (split_ratio = after / before).
- Расчёт скорректированной цены закрытия с учётом сплитов и дивидендов и последующий расчёт лог-доходностей.

Полный алгоритм вычисления скорректированной цены:
1) Загрузить дневные свечи, дивиденды и сплиты, нормализовать индексы и объединить данные в единый DataFrame по датам торгов.
2) Установить столбец split_ratio, где при событии используется \( r_t = \frac{\text{after}_t}{\text{before}_t} \), иначе \( r_t = 1 \).
3) Выполнить обратный проход по датам от самой свежей к более ранним, поддерживая два кумулятивных множителя:
   - \( \text{cum\_split} \leftarrow \text{cum\_split} \cdot r_t \) при наличии события сплита/объединения в дату \( t \).
   - \( \text{cum\_adj} \leftarrow \text{cum\_adj} \cdot \left(1 - \frac{D_t}{\text{cum\_split} \cdot P^{\text{ref}}_t}\right) \) при ненулевом дивиденде \( D_t \).
4) Выбор базовой цены \( P^{\text{ref}}_t \) для дивиденда зависит от даты:
   - Если \( t > \) 2023‑07‑31, то \( P^{\text{ref}}_t \) — цена закрытия предыдущего дня \( P_{t-1} \).
   - Иначе \( P^{\text{ref}}_t \) — цена закрытия за два дня до \( t \), \( P_{t-2} \).
5) Для каждой даты \( t \) скорректированная цена вычисляется как \( \text{adj\_close}_t = P_t \cdot \text{cum\_adj} \).
6) Логарифмические доходности рассчитываются как \( \text{logret}_t = \ln\left(\frac{\text{adj\_close}_t}{\text{adj\_close}_{t-1}}\right) \).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import aiomoex
import numpy as np
import pandas as pd

from finance.moex.moex_common import PrettyPrinter, compute_log_returns


DIV_SWITCH_DATE = pd.to_datetime("2023-07-31")


async def _fetch_board_candles(
    session: aiohttp.ClientSession,
    security: str,
    *,
    engine: str = "stock",
    market: str = "shares",
    board: str = "TQBR",
    interval: int = 24,
) -> pd.DataFrame:
    """
    Загружает дневные свечи по указанной бумаге с ISS MOEX.

    Параметры:
    - session: aiohttp.ClientSession для выполнения запросов
    - security: тикер ценной бумаги (например, 'SBER')
    - engine: движок торговой системы (по умолчанию 'stock')
    - market: рынок (по умолчанию 'shares')
    - board: торговая доска (по умолчанию 'TQBR')
    - interval: интервал свечей
    1 - 1 минута
    10 - 10 минут
    60 - час
    24 - 1 день
    7 - неделя
    31 - месяц
    4 - квартал

    Возвращает:
    - DataFrame с колонками: begin, open, high, low, close, volume, end, value
      и индексом TRADEDATE (дата торгов), отсортированный по возрастанию дат.
      Если данные отсутствуют, возвращает пустой DataFrame.
    """
    candles: List[Dict[str, Any]] = await aiomoex.get_board_candles(
        session,
        security,
        interval=interval,
        board=board,
        market=market,
        engine=engine,
    )
    df = pd.DataFrame(candles)
    if df.empty:
        return df
    df["TRADEDATE"] = pd.to_datetime(df["begin"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"]).set_index("TRADEDATE").sort_index()
    return df


async def _fetch_dividends(
    session: aiohttp.ClientSession,
    security: str,
) -> pd.DataFrame:
    """
    Загрузка денежных дивидендов по бумаге из ISS и парсинг ключевых дат.
    """
    url = f"https://iss.moex.com/iss/securities/{security}/dividends.json"
    async with session.get(url) as resp:
        data: Dict[str, Any] = await resp.json()

    block = data.get("dividends", {})
    cols: List[str] = block.get("columns", []) or []
    rows: List[List[Any]] = block.get("data", []) or []
    df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame()

    if not df.empty:
        for c in ("registryclosedate", "paymentdate"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


async def _fetch_splits(
    session: aiohttp.ClientSession,
    *,
    engine: str,
    security: str,
) -> pd.DataFrame:
    """
    Загрузка сплитов/объединений из ISS и подготовка таблицы для применения коэффициентов.
    """
    url = f"https://iss.moex.com/iss/statistics/engines/{engine}/splits/{security}.json"
    async with session.get(url) as resp:
        data: Dict[str, Any] = await resp.json()

    block = data.get("splits", {}) or {}
    if "columns" in block and "data" in block:
        split_df = pd.DataFrame(block["data"], columns=block["columns"])
    else:
        split_df = pd.DataFrame()

    if not split_df.empty:
        split_df["tradedate"] = pd.to_datetime(split_df["tradedate"], errors="coerce")
        split_df = (
            split_df.dropna(subset=["tradedate", "before", "after"])
            .set_index("tradedate")
            .sort_index()
        )
    return split_df


def _align_dividends_to_price(
    df_price: pd.DataFrame,
    df_divs: pd.DataFrame,
    *,
    align_col: str = "registryclosedate",
) -> pd.Series:
    """
    Выравнивание денежных дивидендов по датам торгового индекса ценового ряда.
    """
    div_ser = pd.Series(0.0, index=df_price.index)
    if (
        df_divs.empty
        or "value" not in df_divs.columns
        or align_col not in df_divs.columns
    ):
        return div_ser

    tmp = df_divs[[align_col, "value"]].dropna().copy()
    tmp.rename(columns={align_col: "date"}, inplace=True)
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"]).set_index("date").sort_index()

    common = df_price.index.intersection(tmp.index)
    if len(common) > 0:
        div_ser.loc[common] = tmp.loc[common, "value"].astype(float)
    return div_ser


def _apply_split_ratios(
    df_price: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.Series:
    """
    Формирование ряда коэффициентов сплита/объединения (split_ratio) по датам торгов.
    """
    ratio = pd.Series(1.0, index=df_price.index)
    if split_df.empty:
        return ratio

    common = df_price.index.intersection(split_df.index)
    r = (split_df.loc[common, "after"] / split_df.loc[common, "before"]).astype(float)
    ratio.loc[common] = r
    return ratio


def _compute_adj_close(
    df: pd.DataFrame,
    *,
    div_switch_date: pd.Timestamp = DIV_SWITCH_DATE,
) -> pd.Series:
    r"""
    Расчёт скорректированной цены закрытия методом обратного накопления корректировок.
    """
    closes = df["close"].to_numpy()[::-1]
    closes = df["open"].to_numpy()[::-1]

    prev_close = df["close"].shift(1).to_numpy()[::-1]
    prev_prev = df["close"].shift(2).to_numpy()[::-1]
    divs = df["dividend"].to_numpy()[::-1]
    splits = df["split_ratio"].to_numpy()[::-1]
    dates = df.index.to_numpy()[::-1]

    adj_closes = closes.copy()
    adj_opens = opens.copy()

    cum_split: float = 1.0
    cum_adj: float = 1.0

    for c, p1, p2, r, d, dt in zip(closes, prev_close, prev_prev, splits, divs, dates):
        cum_split *= float(r)

        if d != 0.0:
            pref = p1 if pd.Timestamp(dt) > div_switch_date else p2
            cum_adj *= 1.0 - float(d) / (cum_split * float(pref))
        adj_rev.append(float(c) * float(cum_adj))

    return pd.Series(adj_rev[::-1], index=df.index, name="adj_close")


def _build_summary_rows(
    security: str,
    df: pd.DataFrame,
    dividend_df: pd.DataFrame,
    split_df: pd.DataFrame,
    fetch_seconds: float,
) -> List[Tuple[str, str, str]]:
    """
    Формирование строк для итоговой сводки: тикер, время fetch, диапазон истории, дивиденды, сплиты.
    """
    first_dt: Optional[pd.Timestamp] = df.index.min()
    last_dt: Optional[pd.Timestamp] = df.index.max()
    n_days = int(df.shape[0])

    div_count = 0
    mean_gap_days = np.nan
    first_div = None
    last_div = None
    if not dividend_df.empty:
        date_col = (
            "paymentdate"
            if "paymentdate" in dividend_df.columns
            else "registryclosedate"
        )
        dates_ser = (
            pd.to_datetime(dividend_df[date_col], errors="coerce")
            .dropna()
            .sort_values()
        )
        div_count = int(len(dates_ser))
        if div_count > 0:
            first_div = dates_ser.iloc[0]
            last_div = dates_ser.iloc[-1]
        if div_count > 1:
            gaps = dates_ser.diff().dropna()
            mean_gap_days = float(gaps.dt.days.mean())

    split_count = 0
    merge_count = 0
    net_ratio = 1.0
    if not split_df.empty:
        after = split_df["after"].astype(float)
        before = split_df["before"].astype(float)
        split_count = int((after > before).sum())
        merge_count = int((after < before).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = (after / before).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratios) > 0:
            net_ratio = float(ratios.prod())

    rows: List[Tuple[str, str, str]] = [
        ("Бумага", security, "magenta"),
        ("Время fetch", f"{fetch_seconds * 1000:.0f} мс", "cyan"),
        (
            "История",
            f"{first_dt.date() if pd.notna(first_dt) else '—'} → {last_dt.date() if pd.notna(last_dt) else '—'}",
            "blue",
        ),
        ("Торговых дней", f"{n_days}", "blue"),
        ("Дивиденды (шт.)", f"{div_count}", "green"),
        (
            "Средний интервал",
            f"{mean_gap_days:.1f} дн." if np.isfinite(mean_gap_days) else "—",
            "green",
        ),
        (
            "Первая выплата",
            f"{first_div.date() if first_div is not None else '—'}",
            "green",
        ),
        (
            "Последняя выплата",
            f"{last_div.date() if last_div is not None else '—'}",
            "green",
        ),
        ("Сплиты", f"{split_count}", "yellow"),
        ("Объединения", f"{merge_count}", "yellow"),
        ("Совокупный коэффициент", f"{net_ratio:.6g}", "yellow"),
    ]
    return rows


async def share_adjusted(
    security: str,
    engine: str = "stock",
    market: str = "shares",
    board: str = "TQBR",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Загрузка цен, дивидендов и событий сплит/объединение по тикеру с ISS MOEX и расчёт adj_close и logret.

    Параметры:
    - security: тикер (SECID), например "SBER".
    - engine: движок ISS (по умолчанию "stock").
    - market: рынок ISS (по умолчанию "shares").
    - board: доска торгов ISS (по умолчанию "TQBR").
    - verbose: если True — печать диагностической информации.

    Возвращает:
    - DataFrame по датам торгов со столбцами:
      open, high, low, close, volume, dividend, split_ratio, adj_close, logret.
    """
    printer = PrettyPrinter(enabled=verbose)

    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        df_price = await _fetch_board_candles(
            session,
            security,
            engine=engine,
            market=market,
            board=board,
        )
        dividend_df = await _fetch_dividends(session, security)
        split_df = await _fetch_splits(session, engine=engine, security=security)
        t1 = time.perf_counter()
        fetch_seconds = t1 - t0

        if df_price.empty:
            if verbose:
                printer.summary_card(
                    f"MOEX {security}",
                    [
                        ("Время fetch", f"{fetch_seconds * 1000:.0f} мс", "cyan"),
                        ("История", "нет данных", "yellow"),
                    ],
                )
            cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "dividend",
                "split_ratio",
                "adj_close",
                "logret",
            ]
            return pd.DataFrame(columns=cols)

        df = df_price.copy()
        df["dividend"] = _align_dividends_to_price(
            df, dividend_df, align_col="registryclosedate"
        )
        df["split_ratio"] = _apply_split_ratios(df, split_df)
        df["adj_close"] = _compute_adj_close(df, div_switch_date=DIV_SWITCH_DATE)
        df["logret"] = compute_log_returns(df["adj_close"])

        if verbose:
            rows = _build_summary_rows(
                security, df, dividend_df, split_df, fetch_seconds
            )
            printer.summary_card(f"MOEX {security}", rows)

        cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dividend",
            "split_ratio",
            "adj_close",
            "logret",
        ]
        available = [c for c in cols if c in df.columns]
        return df[available]
