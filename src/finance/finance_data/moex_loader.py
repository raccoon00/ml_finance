from pathlib import Path

import polars as pl


def load_moex_data(
    data_path: Path,
    date_start: str | None = None,
    date_end: str | None = None,
) -> pl.LazyFrame:
    dfs = []
    files = list(iter(data_path.glob("*")))
    files.sort()
    for file in files:
        date = file.stem[8:]
        date = "-".join([date[:4], date[4:6], date[6:]])

        if date_start is not None and date < date_start:
            continue
        if date_end is not None and date > date_end:
            continue

        df = pl.scan_csv(
            file,
            null_values=["-"],
            schema_overrides={
                "TIME": pl.UInt64,
                "VOLUME": pl.UInt64,
            },
        ).with_columns(
            DATE=pl.lit(date),
            DATETIME=(
                (
                    pl.lit(date) + pl.col("TIME").cast(pl.String).str.zfill(12)
                ).str.strptime(
                    pl.Datetime,
                    format="%Y-%m-%d%H%M%S%f",
                    strict=False,
                    cache=True,
                )
            ),
        )

        dfs.append(df)

    df = pl.concat(dfs)
    return df
