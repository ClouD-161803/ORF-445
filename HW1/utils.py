import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl


# Old implementation
def filter_data_by_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    df['timedelta'] = pd.to_timedelta(df['TIME_M'])
    market_open = pd.Timedelta(hours=9, minutes=30)
    market_close = pd.Timedelta(hours=16, minutes=0) # include 15 minutes after close to capture closing auction
    return df[(df['timedelta'] >= market_open) & (df['timedelta'] < market_close)]


class TAQDataProcessorPandas:
    def __init__(
        self,
        tz_local='America/New_York',
        tz_target='UTC',
    ):
        self.tz_local = tz_local
        self.tz_target = tz_target

    def _apply_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_string = df['DATE'].astype(str) + 'T' + df['TIME_M']
        df['TIMESTAMP'] = pd.to_datetime(dt_string, format='mixed')
        df = df.drop(columns=['DATE', 'TIME_M']).set_index('TIMESTAMP')
        df = df.tz_localize(self.tz_local, ambiguous='infer', nonexistent='shift_forward')
        return df.tz_convert(self.tz_target)

    def save_to_parquet(self, df: pd.DataFrame, output_path: str):
        df = df.sort_index()
        df.to_parquet(output_path, engine='pyarrow', compression='snappy')

    def process_trades(
        self,
        input_path: str,
        output_path: str,
        use_cols: list | None = None,
    ) -> None:
        df = pd.read_csv(input_path, usecols=use_cols, engine='pyarrow')
        df = df[(df['PRICE'] > 0) & (df['SIZE'] > 0)]
        df = self._apply_timestamps(df).sort_index()
        self.save_to_parquet(df, output_path)

    def process_quotes(
        self,
        input_path: str,
        output_path: str,
        use_cols: list | None = None,
    ) -> None:
        df = pd.read_csv(input_path, usecols=use_cols, engine='pyarrow')
        df = df[(df['BID'] > 0) & (df['ASK'] > 0)]
        df = self._apply_timestamps(df).sort_index()
        self.save_to_parquet(df, output_path)

    def filter_by_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.tz_convert(self.tz_local)
        if self.tz_local == 'America/New_York':
            df = df[(df['TIME_M'] >= '09:30:00') & (df['TIME_M'] <= '16:00:00')]
        else:
            print("Filtering by market hours is only supported for EST timezone.")
        return df.tz_convert(self.tz_target)


class TAQDataProcessorPolars:
    def __init__(
        self,
        tz_local='America/New_York',
        tz_target='UTC',
    ):
        self.tz_local = tz_local
        self.tz_target = tz_target

    def _apply_timestamps(
        self, lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Apply timezone-aware timestamp transformation lazily.
        Creates TIMESTAMP column from DATE and TIME_M, then drops originals.
        """
        lf = lf.with_columns(
            pl.concat_str([
                pl.col('DATE').cast(pl.Utf8),
                pl.lit('T'),
                pl.col('TIME_M'),
            ]).alias('TIMESTAMP')
        ).with_columns(
            pl.col('TIMESTAMP').str.to_datetime(time_zone=self.tz_local, exact=False)
            .dt.convert_time_zone(self.tz_target)
            .alias('TIMESTAMP')
        ).drop(['DATE', 'TIME_M'])

        return lf

    def save_to_parquet(
        self, lf: pl.LazyFrame, output_path: str
    ) -> None:
        """Collect lazy frame and save to parquet with sorting."""
        lf = lf.sort('TIMESTAMP')
        lf.collect().write_parquet(
            output_path, compression='snappy'
        )

    def process_trades(
        self,
        input_path: str,
        output_path: str,
        use_cols: list | None = None,
    ) -> None:
        """
        Load, filter, and process trade data lazily.
        Filters for positive PRICE and SIZE.
        """
        lf = pl.scan_csv(input_path)

        if use_cols:
            lf = lf.select(use_cols)

        lf = lf.filter(
            (pl.col('PRICE') > 0) & (pl.col('SIZE') > 0)
        )
        lf = self._apply_timestamps(lf)
        self.save_to_parquet(lf, output_path)

    def process_quotes(
        self,
        input_path: str,
        output_path: str,
        use_cols: list | None = None,
    ) -> None:
        """
        Load, filter, and process quote data lazily.
        Filters for positive BID and ASK.
        """
        lf = pl.scan_csv(input_path)

        if use_cols:
            lf = lf.select(use_cols)

        lf = lf.filter(
            (pl.col('BID') > 0) & (pl.col('ASK') > 0)
        )
        lf = self._apply_timestamps(lf)
        self.save_to_parquet(lf, output_path)

    def filter_by_market_hours(
        self, df: pl.DataFrame
    ) -> pl.DataFrame:
        """Filter data by market hours (EST timezone)."""
        if self.tz_local != 'America/New_York':
            print("Filtering by market hours only supported for EST.")
            return df

        df = df.with_columns(
            pl.col('TIMESTAMP').dt.convert_time_zone(
                self.tz_local
            ).alias('TIMESTAMP_LOCAL')
        )

        market_open = '09:30:00'
        market_close = '16:00:00'

        df = df.filter(
            (pl.col('TIMESTAMP_LOCAL').dt.time() >= market_open)
            & (pl.col('TIMESTAMP_LOCAL').dt.time() <= market_close)
        ).drop('TIMESTAMP_LOCAL')

        return df


