import polars as pl

from finance.resample.aggregations import AggPipeline


def generalized_volume_bars(
    df: pl.LazyFrame,
    volume_column: pl.Expr,
    *,
    volume_per_group: int | float,
    agg_pipeline: AggPipeline,
) -> pl.LazyFrame:
    """
    Args:
        df: dataframe with trading data
        volume_column: column that represents a volume, that should later be groupped
        volume_per_group: if specified as int, rows are groupped into bins with the
            summed volume approximately equal to the specified volume.
            If specifies as a float, first the total volumes of each share is
            calculated, and rows are groupped into specifies fraction of total volume.
        agg_pipeline: the resulting groups are aggregated with a specified pipeline.
    """
    df = df.with_columns(
        volume_column.cum_sum()
        .over("SECCODE", "DATE", order_by="TIME")
        .alias("vol_cumsum")
    )

    if isinstance(volume_per_group, float):
        null_filter = pl.col("PRICE").is_null() | pl.col("VOLUME").is_null()
        date_sec_volume = (
            pl.col("vol_cumsum")
            .over("SECCODE", "DATE", order_by="TIME")
            .filter(~null_filter)
            .last()
        )
        df = df.with_columns(
            (date_sec_volume * volume_per_group).alias("volume_per_group")
        )
    else:
        df = df.with_columns(pl.lit(volume_per_group).alias("volume_per_group"))

    group_col = pl.col("vol_cumsum").floordiv(pl.col("volume_per_group"))
    df_aggd = (
        df.with_columns(group_col.alias("grp"))
        .group_by("DATE", "SECCODE", "grp")
        .agg(*agg_pipeline.apply())
        .drop("grp")  # volume_per_group and vol_cumsum are dropped in agg
    )

    return df_aggd


def volume_bars(
    df: pl.LazyFrame,
    *,
    volume_per_group: int | float = 1e-4,
    agg_pipeline: AggPipeline = AggPipeline.moex(),
) -> pl.LazyFrame:
    """
    Args:
        df: dataframe with trading data
        volume_per_group: if specified as int, rows are groupped into bins with the
            summed volume approximately equal to the specified volume.
            If specifies as a float, first the total volumes of each share is
            calculated, and rows are groupped into specifies fraction of total volume.
        agg_pipeline: the resulting groups are aggregated with a specified pipeline.
    """
    trades = pl.col("TRADENO").is_not_null()
    volume_column = pl.when(trades).then(pl.col("VOLUME")).otherwise(0)
    return generalized_volume_bars(
        df=df,
        volume_column=volume_column,
        volume_per_group=volume_per_group,
        agg_pipeline=agg_pipeline,
    )


def tick_bars(
    df: pl.LazyFrame,
    *,
    volume_per_group: int | float = 1e-4,
    agg_pipeline: AggPipeline = AggPipeline.moex(),
) -> pl.LazyFrame:
    """
    Args:
        df: dataframe with trading data
        volume_per_group: if specified as int, rows are groupped into bins with the
            summed volume approximately equal to the specified volume.
            If specifies as a float, first the total volumes of each share is
            calculated, and rows are groupped into specifies fraction of total volume.
        agg_pipeline: the resulting groups are aggregated with a specified pipeline.
    """
    trades = pl.col("TRADENO").is_not_null()
    idx = pl.lit(1).cum_count().over("SECCODE", "DATE", order_by="TIME")
    volume_column = pl.when(trades).then(idx).otherwise(0)
    return generalized_volume_bars(
        df=df,
        volume_column=volume_column,
        volume_per_group=volume_per_group,
        agg_pipeline=agg_pipeline,
    )


def currency_bars(
    df: pl.LazyFrame,
    *,
    volume_per_group: int | float = 1e-4,
    agg_pipeline: AggPipeline = AggPipeline.moex(),
) -> pl.LazyFrame:
    """
    Args:
        df: dataframe with trading data
        volume_per_group: if specified as int, rows are groupped into bins with the
            summed volume approximately equal to the specified volume.
            If specifies as a float, first the total volumes of each share is
            calculated, and rows are groupped into specifies fraction of total volume.
        agg_pipeline: the resulting groups are aggregated with a specified pipeline.
    """
    trades = pl.col("TRADENO").is_not_null()
    trade_volume = pl.when(trades).then(pl.col("VOLUME")).otherwise(0)
    volume_column = trade_volume * pl.col("PRICE")
    return generalized_volume_bars(
        df=df,
        volume_column=volume_column,
        volume_per_group=volume_per_group,
        agg_pipeline=agg_pipeline,
    )
