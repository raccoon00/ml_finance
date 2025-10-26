import pandas as pd
from typing import Optional

# Based on MOEX index structure
MOEX_INDICES = {
    "broad_market": {
        "IMOEX": "Индекс МосБиржи (рубли)",
        "IMOEX10": "Индекс МосБиржи 10",
        "RTSI": "RTS Index (доллары США)",
        "IMOEX2": "Индекс МосБиржи 2 (расширенный рынок)"
    },
    "sectoral": {
        "MOEXOG": "Нефть и газ",
        "MOEXFN": "Финансовый сектор",
        "MOEXCN": "Потребительский сектор",
        "MOEXMM": "Металлы и добыча",
        "MOEXCH": "Химия и нефтехимия",
        "MOEXTN": "Телекоммуникации",
        "MOEXTR": "Транспорт",
        "MOEXEU": "Энергетика",
        "MOEXIN": "Промышленность",
        "MOEXRE": "Недвижимость"
    },
    "bonds": {
        "RUGBITR": "Индекс государственных облигаций RGBI Total Return",
        "RUGBCP": "Корпоративные облигации — общий индекс",
        "RUGBTR": "Индекс корпоративных облигаций полной доходности",
        "RUGBMB": "Индекс муниципальных облигаций"
    },
    "currency": {
        "USDCBX": "USD/RUB индикативный курс",
        "EURCBX": "EUR/RUB индикативный курс",
        "CNYCBX": "CNY/RUB индикативный курс",
        "USDRUB_TOM": "USD/RUB завтра (T+1)"
    },
    "custom_groups": {
        "sustainable": {
            "IMOEXESG": "Индекс МосБиржи ESG",
            "IMOEXBC": "Индекс МосБиржи устойчивого развития"
        },
        "innovation": {
            "MOEXINN": "Индекс инновационных компаний",
            "MOEXIBB": "Индекс интернет-компаний"
        }
    }
}

def get_df() -> pd.DataFrame:
    """Returns pandas DataFrame with columns: index, desc, group"""
    data = []

    for group, indices in MOEX_INDICES.items():
        if isinstance(indices, dict):
            for idx, desc in indices.items():
                if isinstance(desc, dict):
                    for sub_idx, sub_desc in desc.items():
                        data.append([sub_idx, sub_desc, f"{group}/{idx}"])
                else:
                    data.append([idx, desc, group])

    return pd.DataFrame(data, columns=["index", "desc", "group"])

def describe(*, index: Optional[str] = None, group: Optional[str] = None) -> str:
    df = get_df()

    if index:
        row = df[df["index"] == index]
        if row.empty:
            return f"Индекс '{index}' не найден."
        code, desc, grp = row.iloc[0]
        return f"{code} — {desc} (группа: {grp})"

    elif group:
        subset = df[df["group"].str.contains(group, case=False, na=False)]
        if subset.empty:
            return f"Группа '{group}' не найдена."
        w_index = subset['index'].str.len().max()
        w_desc = subset['desc'].str.len().max()
        output = [f"Группа: {group}"]
        output.append(f"  {'Индекс':<{w_index}} | {'Описание':<{w_desc}}")
        output.append(f"  {'-'*w_index}-+-{'-'*w_desc}")
        for _, row in subset.iterrows():
            output.append(f"  {row['index']:<{w_index}} | {row['desc']:<{w_desc}}")
        return "\n".join(output)

    else:
        w_index = df['index'].str.len().max()
        w_desc = df['desc'].str.len().max()
        grouped = df.groupby("group")

        output = []
        for grp, rows in grouped:
            output.append(f"Группа: {grp}")
            output.append(f"  {'Индекс':<{w_index}} | {'Описание':<{w_desc}}")
            output.append(f"  {'-'*(w_index)}-+-{'-'*(w_desc)}")
            for _, row in rows.iterrows():
                output.append(f"  {row['index']:<{w_index}} | {row['desc']:<{w_desc}}")
            output.append("")
        return "\n".join(output)

