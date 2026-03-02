import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from zoneinfo import ZoneInfo


EXCHANGE_NAMES = {
    'A': 'NYSE American', 'B': 'NASDAQ BX', 'C': 'NSX',
    'D': 'FINRA ADF', 'H': 'MIAX', 'J': 'Cboe EDGA',
    'K': 'Cboe EDGX', 'M': 'CHX', 'N': 'NYSE',
    'P': 'NYSE Arca', 'Q': 'NASDAQ', 'U': 'MEMX',
    'V': 'IEX', 'X': 'NASDAQ PSX', 'Y': 'Cboe BYX',
    'Z': 'Cboe BZX',
}


# Old implementation
def filter_data_by_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter market data by trading hours (9:30 AM - 4:00 PM EST)."""
    # Assume data has TIMESTAMP column from Polars processor
    # Convert to local timezone (EST) to extract hour of day
    ts = pd.to_datetime(df['TIMESTAMP'])
    ts_local = ts.dt.tz_convert('America/New_York')

    # Extract hour and minute to compare against market hours (9:30 - 16:00)
    market_open = ts_local.dt.hour * 3600 + ts_local.dt.minute * 60 + ts_local.dt.second
    market_open_seconds = 9 * 3600 + 30 * 60  # 09:30:00
    market_close_seconds = 16 * 3600  # 16:00:00

    return df[(market_open >= market_open_seconds) & (market_open < market_close_seconds)]





class TAQDataProcessorPandas:
    def __init__(
        self,
        tz_local='America/New_York',
        tz_target='UTC',
    ):
        self.tz_local = tz_local
        self.tz_target = tz_target

    def _apply_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        # Data is in DD/MM/YYYY format from WRDS
        dt_string = df['DATE'].astype(str) + 'T' + df['TIME_M']
        # %f only captures 6 digits (microseconds), but TIME_M has 9 (nanoseconds)
        # So parse base datetime, then add nanoseconds separately
        df['TIMESTAMP'] = pd.to_datetime(dt_string, format='%d/%m/%YT%H:%M:%S.%f')

        # Extract full nanosecond precision from TIME_M fractional part
        time_m_str = df['TIME_M'].astype(str)
        fractional_part = time_m_str.str.split('.').str[1]  # Get digits after decimal
        # Pad/truncate to 9 digits and convert to nanoseconds
        ns_offsets = fractional_part.apply(lambda x: int(x.ljust(9, '0')[:9]))
        df['TIMESTAMP'] = df['TIMESTAMP'] + pd.to_timedelta(ns_offsets, unit='ns')

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
        Uses exact='ns' to preserve nanosecond precision from the source data.
        """
        lf = lf.with_columns(
            pl.concat_str([
                pl.col('DATE').cast(pl.Utf8),
                pl.lit('T'),
                pl.col('TIME_M'),
            ]).alias('TIMESTAMP')
        ).with_columns(
            pl.col('TIMESTAMP').str.to_datetime(time_zone=self.tz_local, time_unit='ns')
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


class NBBO:
    """
    National Best Bid and Offer computation and visualization.

    Computes the NBBO from multi-exchange TAQ quote data:
      NBBO Bid = max(prevailing bid across all valid exchanges)
      NBBO Ask = min(prevailing ask across all valid exchanges)

    The prevailing quote from each exchange at time t is defined as the
    most recent quote published by that exchange at or before t (last
    observation carried forward). This is implemented efficiently via
    pandas merge_asof with direction='backward', fully exploiting
    nanosecond-precision timestamps.
    """

    def __init__(
        self,
        quotes: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        exclude_exchanges: list[str] | None = None,
        min_records: int = 100,
    ):
        self.exclude_exchanges = exclude_exchanges or []
        self.min_records = min_records

        # Determine valid exchanges
        counts = quotes['EX'].value_counts()
        valid = counts[counts >= min_records].index.tolist()
        self.valid_exchanges = sorted(
            ex for ex in valid if ex not in self.exclude_exchanges
        )

        # Store filtered, sorted quotes
        self.quotes = (
            quotes[quotes['EX'].isin(self.valid_exchanges)]
            .sort_values(by='TIMESTAMP')
            .reset_index(drop=True)
        )
        self.trades = (
            trades.sort_values(by='TIMESTAMP').reset_index(drop=True)
            if trades is not None else None
        )

        # Cached results
        self._nbbo_series: pd.DataFrame | None = None
        self._trades_with_nbbo: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Compute the NBBO time series over a window.

        For every unique quote timestamp in [start, end], determines the
        prevailing quote from each exchange (merge_asof backward) and
        returns NBBO_BID (max), NBBO_ASK (min), NBBO_MID, NBBO_SPREAD.
        """
        quotes = self.quotes
        if start is not None:
            quotes = quotes[quotes['TIMESTAMP'] >= start]
        if end is not None:
            quotes = quotes[quotes['TIMESTAMP'] <= end]

        if quotes.empty:
            self._nbbo_series = pd.DataFrame(
                columns=['TIMESTAMP', 'NBBO_BID', 'NBBO_ASK',
                         'NBBO_MID', 'NBBO_SPREAD']
            )
            return self._nbbo_series

        # Unified timestamp grid
        all_times = (
            quotes['TIMESTAMP']
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        result = pd.DataFrame({'TIMESTAMP': all_times})

        bid_cols, ask_cols = [], []

        for ex in self.valid_exchanges:
            ex_q = (
                quotes[quotes['EX'] == ex][['TIMESTAMP', 'BID', 'ASK']]
                .sort_values(by='TIMESTAMP')
                .reset_index(drop=True)
            )
            if ex_q.empty:
                continue

            merged = pd.merge_asof(
                result[['TIMESTAMP']], ex_q,
                on='TIMESTAMP', direction='backward',
            )
            result[f'BID_{ex}'] = merged['BID']
            result[f'ASK_{ex}'] = merged['ASK']
            bid_cols.append(f'BID_{ex}')
            ask_cols.append(f'ASK_{ex}')

        result['NBBO_BID'] = result[bid_cols].max(axis=1)
        result['NBBO_ASK'] = result[ask_cols].min(axis=1)
        result['NBBO_MID'] = (result['NBBO_BID'] + result['NBBO_ASK']) / 2
        result['NBBO_SPREAD'] = result['NBBO_ASK'] - result['NBBO_BID']

        clean = result[
            ['TIMESTAMP', 'NBBO_BID', 'NBBO_ASK', 'NBBO_MID', 'NBBO_SPREAD']
        ].copy()
        self._nbbo_series = clean
        return clean

    def compute_at_trades(self) -> pd.DataFrame:
        """
        Compute NBBO at each trade timestamp via merge_asof.

        For each exchange, finds the most recent quote at or before each
        trade time, then takes NBBO_BID = max(bids), NBBO_ASK = min(asks).
        Also records which exchange provided the best bid/ask.
        """
        if self.trades is None:
            raise ValueError("No trades data provided.")

        trades = self.trades.copy()

        bids, asks = {}, {}
        for ex in self.valid_exchanges:
            ex_q = (
                self.quotes[self.quotes['EX'] == ex][['TIMESTAMP', 'BID', 'ASK']]
                .sort_values(by='TIMESTAMP')
                .reset_index(drop=True)
            )
            if ex_q.empty:
                continue

            merged = pd.merge_asof(
                trades[['TIMESTAMP']], ex_q,
                on='TIMESTAMP', direction='backward',
            )
            bids[ex] = merged['BID'].values
            asks[ex] = merged['ASK'].values

        bids_df = pd.DataFrame(bids, index=trades.index)
        asks_df = pd.DataFrame(asks, index=trades.index)

        trades['NBBO_BID'] = bids_df.max(axis=1, skipna=True)
        trades['NBBO_ASK'] = asks_df.min(axis=1, skipna=True)

        # idxmax / idxmin raise on all-NaN rows; guard with a mask
        has_bid = bids_df.notna().any(axis=1)
        has_ask = asks_df.notna().any(axis=1)
        trades['NBBO_BID_EX'] = pd.Series('', index=trades.index, dtype='object')
        trades['NBBO_ASK_EX'] = pd.Series('', index=trades.index, dtype='object')
        if has_bid.any():
            trades.loc[has_bid, 'NBBO_BID_EX'] = (
                bids_df.loc[has_bid].idxmax(axis=1).values
            )
        if has_ask.any():
            trades.loc[has_ask, 'NBBO_ASK_EX'] = (
                asks_df.loc[has_ask].idxmin(axis=1).values
            )
        trades['NBBO_MID'] = (trades['NBBO_BID'] + trades['NBBO_ASK']) / 2
        trades['NBBO_SPREAD'] = trades['NBBO_ASK'] - trades['NBBO_BID']

        self._trades_with_nbbo = trades
        return trades

    # ------------------------------------------------------------------
    # Trade classification
    # ------------------------------------------------------------------

    def classify_trades(self, tolerance: float = 1e-6) -> pd.DataFrame:
        """
        Classify each trade relative to the NBBO.

        Categories: below_bid, at_bid, between, at_ask, above_ask.
        """
        if self._trades_with_nbbo is None:
            self.compute_at_trades()

        df = self._trades_with_nbbo
        assert df is not None

        conditions = [
            df['PRICE'] < df['NBBO_BID'] - tolerance,
            (df['PRICE'] >= df['NBBO_BID'] - tolerance)
            & (df['PRICE'] <= df['NBBO_BID'] + tolerance),
            (df['PRICE'] > df['NBBO_BID'] + tolerance)
            & (df['PRICE'] < df['NBBO_ASK'] - tolerance),
            (df['PRICE'] >= df['NBBO_ASK'] - tolerance)
            & (df['PRICE'] <= df['NBBO_ASK'] + tolerance),
            df['PRICE'] > df['NBBO_ASK'] + tolerance,
        ]
        choices = ['below_bid', 'at_bid', 'between', 'at_ask', 'above_ask']
        df['TRADE_CLASS'] = np.select(conditions, choices, default='unknown')

        self._trades_with_nbbo = df
        return df

    def summary(self) -> pd.DataFrame:
        """Print and return a summary of trade classification statistics."""
        if (self._trades_with_nbbo is None
                or 'TRADE_CLASS' not in self._trades_with_nbbo.columns):
            self.classify_trades()

        df = self._trades_with_nbbo
        assert df is not None
        total = len(df)

        order = ['below_bid', 'at_bid', 'between', 'at_ask', 'above_ask']
        labels = [
            'Below Bid', 'At Bid', 'Between Bid-Ask',
            'At Ask', 'Above Ask',
        ]

        rows = []
        for cat, label in zip(order, labels):
            count = int((df['TRADE_CLASS'] == cat).sum())
            rows.append({
                'Category': label,
                'Count': count,
                'Fraction': count / total if total else 0,
            })

        summary_df = pd.DataFrame(rows)

        print(f"\nTrade Classification Summary (n={total:,})")
        print(f"{'Category':<20} {'Count':>8} {'Fraction':>10}")
        print('-' * 40)
        for _, row in summary_df.iterrows():
            print(
                f"{row['Category']:<20} "
                f"{row['Count']:>8,} "
                f"{row['Fraction']:>10.2%}"
            )

        inside = int(summary_df.loc[
            summary_df['Category'].isin(
                ['At Bid', 'Between Bid-Ask', 'At Ask']
            ), 'Count'
        ].sum())
        outside = int(summary_df.loc[
            summary_df['Category'].isin(
                ['Below Bid', 'Above Ask']
            ), 'Count'
        ].sum())
        print(f"\n{'Inside NBBO':<20} {inside:>8,} {inside / total:>10.2%}")
        print(f"{'Outside NBBO':<20} {outside:>8,} {outside / total:>10.2%}")

        inverted = int((df['NBBO_SPREAD'] < 0).sum())
        print(f"\nInverted spreads: {inverted:,} ({inverted / total:.2%})")
        print(f"Mean NBBO spread:   ${df['NBBO_SPREAD'].mean():.4f}")
        print(f"Median NBBO spread: ${df['NBBO_SPREAD'].median():.4f}")

        return summary_df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        title: str = 'NBBO Analysis',
        figsize: tuple[float, float] = (18, 10),
        save_path: str | None = None,
        quote_alpha: float = 0.35,
        quote_linewidth: float = 0.6,
        trade_size: float = 20,
        nbbo_alpha: float = 0.25,
        show_legend: bool = True,
        max_points_per_series: int = 5000,
        tz_display: str = 'America/New_York',
    ) -> Figure:
        """
        Elaborate NBBO plot with:
          - Per-exchange quotes as coloured step lines
          - Trades as scatter dots (green = inside NBBO, red = outside)
          - NBBO region shaded grey with solid boundary lines
          - NBBO spread subplot below the main price chart
        """
        # Compute NBBO for window
        nbbo = self.compute(start=start, end=end)
        if nbbo.empty:
            raise ValueError("No data in the specified time window.")

        # Filter quotes to window
        quotes_w = self.quotes
        if start is not None:
            quotes_w = quotes_w[quotes_w['TIMESTAMP'] >= start]
        if end is not None:
            quotes_w = quotes_w[quotes_w['TIMESTAMP'] <= end]

        # Filter trades to window
        trades_w = None
        source = self._trades_with_nbbo if self._trades_with_nbbo is not None else self.trades
        if source is not None:
            trades_w = source.copy()
            if start is not None:
                trades_w = trades_w[trades_w['TIMESTAMP'] >= start]
            if end is not None:
                trades_w = trades_w[trades_w['TIMESTAMP'] <= end]

        # Downsample NBBO for plotting
        def _downsample(frame: pd.DataFrame, max_pts: int) -> pd.DataFrame:
            if len(frame) > max_pts:
                step = max(1, len(frame) // max_pts)
                return frame.iloc[::step]
            return frame

        nbbo_p = _downsample(nbbo, max_points_per_series)

        # ---- Figure setup ----
        fig, (ax, ax_s) = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
        fig.patch.set_facecolor('white')

        for a in (ax, ax_s):
            a.set_facecolor('#fafafa')
            a.grid(True, alpha=0.25, linestyle='--', color='grey')
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

        # ---- Exchange colours ----
        cmap = plt.colormaps['tab20']
        ex_colors = {
            ex: cmap(i) for i, ex in enumerate(self.valid_exchanges)
        }

        # ---- Exchange quote step lines ----
        for ex in self.valid_exchanges:
            ex_data = (
                quotes_w[quotes_w['EX'] == ex].sort_values(by='TIMESTAMP')
            )
            if ex_data.empty:
                continue
            ex_data = _downsample(ex_data, max_points_per_series)

            color = ex_colors[ex]
            name = EXCHANGE_NAMES.get(ex, ex)

            ax.step(
                ex_data['TIMESTAMP'], ex_data['BID'],
                color=color, alpha=quote_alpha,
                linewidth=quote_linewidth,
                where='post', label=name,
            )
            ax.step(
                ex_data['TIMESTAMP'], ex_data['ASK'],
                color=color, alpha=quote_alpha,
                linewidth=quote_linewidth,
                where='post',
            )

        # ---- NBBO shading ----
        ax.fill_between(
            nbbo_p['TIMESTAMP'],
            nbbo_p['NBBO_BID'],
            nbbo_p['NBBO_ASK'],
            alpha=nbbo_alpha, color='silver',
            label='NBBO Region', step='post',
            zorder=2, edgecolor='none',
        )

        # ---- NBBO boundary lines ----
        ax.step(
            nbbo_p['TIMESTAMP'], nbbo_p['NBBO_BID'],
            color='black', linewidth=1.3, alpha=0.85,
            where='post', label='NBBO Bid', zorder=3,
        )
        ax.step(
            nbbo_p['TIMESTAMP'], nbbo_p['NBBO_ASK'],
            color='black', linewidth=1.3, alpha=0.85,
            where='post', label='NBBO Ask',
            linestyle='--', zorder=3,
        )

        # ---- Trades ----
        if trades_w is not None and not trades_w.empty:
            if 'TRADE_CLASS' in trades_w.columns:
                inside = trades_w['TRADE_CLASS'].isin(
                    ['at_bid', 'between', 'at_ask']
                )
                t_in = trades_w[inside]
                t_out = trades_w[~inside]
                if not t_in.empty:
                    ax.scatter(
                        t_in['TIMESTAMP'], t_in['PRICE'],
                        s=trade_size, c='#2ecc71', alpha=0.75,
                        zorder=5, label='Trades (inside NBBO)',
                        edgecolors='black', linewidths=0.3,
                    )
                if not t_out.empty:
                    ax.scatter(
                        t_out['TIMESTAMP'], t_out['PRICE'],
                        s=trade_size, c='#e74c3c', alpha=0.75,
                        zorder=5, label='Trades (outside NBBO)',
                        edgecolors='black', linewidths=0.3,
                    )
            else:
                ax.scatter(
                    trades_w['TIMESTAMP'], trades_w['PRICE'],
                    s=trade_size, c='#e74c3c', alpha=0.75,
                    zorder=5, label='Trades',
                    edgecolors='black', linewidths=0.3,
                )

        # ---- Labels & legend (main) ----
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        if show_legend:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.01, 1),
                fontsize=8, framealpha=0.95, edgecolor='grey',
            )

        # ---- Spread subplot ----
        ax_s.step(
            nbbo_p['TIMESTAMP'], nbbo_p['NBBO_SPREAD'],
            color='steelblue', linewidth=0.8, where='post',
        )
        ax_s.fill_between(
            nbbo_p['TIMESTAMP'], 0, nbbo_p['NBBO_SPREAD'],
            alpha=0.3, color='steelblue', step='post',
        )
        ax_s.axhline(
            y=0, color='red', linewidth=0.5,
            linestyle='--', alpha=0.5,
        )
        ax_s.set_ylabel('Spread ($)', fontsize=11, fontweight='bold')
        ax_s.set_xlabel('Time', fontsize=12, fontweight='bold')

        # ---- X-axis date formatting ----
        tz = ZoneInfo(tz_display) if tz_display else None
        ax_s.xaxis.set_major_formatter(
            mdates.DateFormatter('%H:%M', tz=tz)
        )
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        return fig
