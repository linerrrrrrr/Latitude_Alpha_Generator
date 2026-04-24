import numpy as np
import pandas as pd


def f_0107(bars: pd.DataFrame, f_name: str = 'f_0107', roll_days: int = 20) -> pd.DataFrame:
    """
    factor_intro: 极端收益反转因子
    category: high_frequency
    category_intro: 高频因子
    subcategory: momentum_reversal
    subcategory_intro: 动量反转类因子
    min_period: 20d
    source: 魏建炜, 盛少成, 苏恩, 2022, 日内极端收益前后的反转特性与因子构建, 开源证券
    author: 因子团队

    说明：基于股票i的分钟行情数据，每日寻找出日内收益最极端的那根bar，
    即S最大的那根bar：S = |ri,t - median({ri,t})|，其中 ri,t 为分钟收益率（close 的 pct_change），
    median({ri,t}) 为当日所有分钟收益率的中位数。每日取 S 最大那根 bar 的收益率 ri,tmax(S)
    及其前一分钟收益率 ri,tmax(S)-1；研报为每月底将过去20天两者分别求均后截面 rank 等权求和。
    本实现采用日频、过去 roll_days 日滚动均值替代月频，即
    rank( (1/20)*sum(ri,tmax(S)) ) + rank( (1/20)*sum(ri,tmax(S)-1) )。
    日内参与 S 与 median 的为当日有效 ret 的分钟（首分钟 pct_change 为 NaN 已排除）。
    日内最极端收益呈反转特性，最极端收益后呈动量特性；该因子具有显著反转效应。

    Parameters
    ----------
    bars: pd.DataFrame
        分钟频, bars
        ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    f_name: str
        因子名称
    roll_days: int
        滚动窗口天数，默认为 20

    Returns
    -------
    factors: pd.DataFrame
        日频, factors
        ['datetime', 'symbol', 'factor_name', 'factor_value']
    """
    required_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = set(required_cols) - set(bars.columns)
    if missing_cols:
        raise ValueError(f"缺失必要字段: {', '.join(sorted(missing_cols))}")

    # 分钟频：先建 date，再算收益率与日内 S
    df = (
        bars.loc[:, required_cols]
        .copy()
        .assign(datetime=lambda x: pd.to_datetime(x['datetime'], errors='coerce'))
        .sort_values(['symbol', 'datetime'])
        .assign(date=lambda x: x['datetime'].dt.date)
        .assign(ret=lambda x: x.groupby('symbol')['close'].pct_change())
    )

    # 每日：找 S 最大的 bar，得到该 bar 的收益率及前一分钟收益率（研报 ri,tmax(S) 与 ri,tmax(S)-1）
    def _daily_ret_extreme(samples: pd.DataFrame) -> pd.Series:
        samples = samples.dropna(subset=['ret'])
        if len(samples) == 0:
            return pd.Series({'ret_extreme': np.nan, 'ret_pre_extreme': np.nan})
        s_val = (samples['ret'] - samples['ret'].median()).abs()
        max_pos = s_val.values.argmax()
        ret_extreme = samples['ret'].iloc[max_pos]
        ret_pre_extreme = samples['ret'].iloc[max_pos - 1] if max_pos > 0 else np.nan
        return pd.Series({'ret_extreme': ret_extreme, 'ret_pre_extreme': ret_pre_extreme})

    daily_df = (
        df.groupby(['date', 'symbol'], as_index=False)
        .apply(_daily_ret_extreme)
        .rename(columns={'date': 'datetime'})
        .sort_values(['symbol', 'datetime'])
    )

    # 日频：过去 roll_days 日滚动均值，截面 rank，等权求和
    # 按 symbol 分组，计算时序的滚动均值
    symbol_gby = daily_df.groupby('symbol')
    daily_df['extreme_ret'] = symbol_gby['ret_extreme'].transform(
        lambda x: x.rolling(roll_days, min_periods=5).mean())
    daily_df['extreme_before_ret'] = symbol_gby['ret_pre_extreme'].transform(
        lambda x: x.rolling(roll_days, min_periods=5).mean())
    # 按 datetime 分组，计算截面的rank
    datetime_gby = daily_df.groupby('datetime')
    daily_df['extreme_rank'] = datetime_gby['extreme_ret'].rank()
    daily_df['extreme_before_rank'] = datetime_gby['extreme_before_ret'].rank()
    # 等权求和
    daily_df['factor_value'] = daily_df['extreme_rank'] + daily_df['extreme_before_rank']
    daily_df['factor_name'] = f_name
    daily_df['datetime'] = pd.to_datetime(daily_df['datetime'])

    return daily_df[['datetime', 'symbol', 'factor_name', 'factor_value']]


if __name__ == "__main__":

    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    _root = next((p for p in _here.parents if (p / "factor_script").is_dir()), _here)
    sys.path.insert(0, str(_root))

    import zenidatasdk as zd

    try:
        import config_local as config
    except ImportError:
        import config

    client = zd.Client(
        base_url=config.ZENI_URL,
        username=config.ZENI_USERNAME,
        password=config.ZENI_PASSWORD,
    )
    init_date = "2025-07-01"
    end_date = "2025-09-30"
    index_symbol = "000905.XSHG"

    print("获取指数成分股...")
    index_weights_df = client.get_index_constituents_df(
        index_symbol=index_symbol,
        start_date=init_date,
        end_date=end_date,
    )
    symbols = index_weights_df["symbol"].unique().tolist()

    print("获取分钟K线数据...")
    bars = client.get_kline_df(
        symbol=symbols,
        start_date=init_date,
        end_date=end_date,
        frequency="1m",
        adjust_type="post",
        market="cn_stock",
    )

    print("计算因子...")
    factor = f_0107(bars, f_name="f_0107")
    print(f"因子计算完成:\n{factor.info()}\n{factor.tail()}")