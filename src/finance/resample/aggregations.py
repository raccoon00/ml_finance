from typing import Any, Callable

import polars as pl


class AggPipeline:
    def __init__(self):
        self._init_cols: dict[str, pl.Expr] = None
        self._dirs: dict[str, pl.Expr] = None
        self._stats: dict[str, dict[str, Any]] = None

    def init_stats_default(self) -> "AggPipeline":
        percentiles = {
            f"P{int(p * 100):02d}": lambda c: c.quantile(p)
            for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        }
        central_moments = {
            f"S{m:d}": lambda c: ((c - c.mean()) ** m).mean() for m in range(2, 6)
        }
        statistics = dict(
            all=dict(
                mean=lambda c: c.mean(),
                min=lambda c: c.min(),
                max=lambda c: c.max(),
                range=lambda c: c.max() - c.min(),
                first=lambda c: c.first(),
                last=lambda c: c.last(),
            ),
            time=dict(),
            datetime=dict(),
            volume=dict(
                sum=lambda c: c.sum(),
                **percentiles,
                **central_moments,
            ),
            price=dict(
                **percentiles,
                **central_moments,
            ),
            total_price=dict(
                sum=lambda c: c.sum(),
                **percentiles,
                **central_moments,
            ),
        )
        self._stats = statistics
        return self

    @classmethod
    def moex(cls) -> "AggPipeline":
        pipe = (
            cls()
            .init_columns_moex()
            .init_action_types_moex()
            .init_directions_moex()
            .init_stats_default()
        )
        return pipe

    def init_columns_moex(self) -> "AggPipeline":
        self._init_cols = dict(
            time=pl.col("TIME"),
            datetime=pl.col("DATETIME"),
            volume=pl.col("VOLUME"),
            price=pl.col("PRICE"),
            total_price=pl.col("PRICE") * pl.col("VOLUME"),
        )
        return self

    def init_directions_moex(self) -> "AggPipeline":
        self._dirs = dict(
            buy=pl.col("BUYSELL") == "B",
            sell=pl.col("BUYSELL") == "S",
            both=pl.lit(True),
        )
        return self

    def init_action_types_moex(self) -> "AggPipeline":
        self._action_types = dict(
            order=pl.col("TRADENO").is_not_null(),
            book=pl.col("TRADENO").is_null(),
            both=pl.lit(True),
        )
        return self

    def apply(self) -> list[pl.Expr]:
        init_cols = self._init_cols
        dirs = self._dirs
        types = self._action_types
        stats: dict[str, dict[str, Callable[[pl.Expr], pl.Expr]]] = self._stats

        agg_cols: list[pl.Expr] = []
        for name, col in init_cols.items():
            aggs = dict()
            aggs.update(stats["all"])
            aggs.update(stats[name])
            for agg_name, agg in aggs.items():
                for dir_name, dir_filter in dirs.items():
                    for type_name, type_filter in types.items():
                        col_name = (
                            agg_name + "_" + dir_name + "_" + type_name + "_" + name
                        )
                        agg_cols.append(
                            agg(col.filter(dir_filter & type_filter)).alias(col_name)
                        )
        return agg_cols
