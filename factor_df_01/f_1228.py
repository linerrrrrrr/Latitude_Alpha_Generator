import numpy as np
import pandas as pd


def f_1228(bars: pd.DataFrame, f_name: str = 'Average_relative_price_position', roll_days: int = 20) -> pd.DataFrame:
    """
    factor_intro: 衡量股票在价格相对高位停留的时间长短
    category: 高频因子
    category_intro: 收益率分布类
    subcategory:
    subcategory_intro:
    min_period: 20d
    source: 朱剑涛, 2020, 基于时间维度度量的日内买卖压力, 东方证券
    author: 因子团队

    说明：使用每日K线的典型价格（(开盘+收盘+最高+最低)/4）计算日内时间加权平均价格（TWAP），
    并在过去 cal_range 个交易日内按交易时长加权平均，再对过去 roll_days 个交易日做滚动平滑，
    作为原始因子值。根据以上计算过程，判断因子方向为正向。
    因子逻辑：股票在价格相对高位停留的时间越长，表明买方压力越大，未来收益越高；
    通过度量价格在相对高位停留的时间长短，捕捉买方压力较大的股票。

    Parameters
    ----------
    bars: pd.DataFrame
        分钟频 bars
        ['datetime', 'symbol', 'open', 'high', 'low', 'close']
    f_name: str
        因子名称，默认为 'Average_relative_price_position'
    cal_range: int
        时间窗口天数，默认为 1。研报原文分别基于1日、5日、20日三个时间窗口
        滚动计算指标，本实现采用过去 cal_range 日滚动加权平均替代。
    roll_days: int
        滚动平滑天数，默认为 20。研报原文做20个交易日的平滑得到最终因子，
        本实现采用过去 roll_days 日滚动均值替代。

    Returns
    -------
    factors: pd.DataFrame
        日频 factors
        ['datetime', 'symbol', 'factor_name', 'factor_value']
    """
    # pass
    # bars = bars_1m_df.copy()
    # f_name = 'Average_relative_price_position'
    # # cal_range = 1
    # roll_days = 20

    bars = bars.rename(columns = {'datetime': 'datetime_min'})
    bars['datetime'] = bars['datetime_min'].dt.normalize()

    bars['TWAP_min'] = (bars['open'] + bars['high'] + bars['low'] + bars['close']) / 4
    bars = bars.groupby(['symbol', 'datetime']).agg(
        TWAP_day = ('TWAP_min', 'mean'),
        num_of_Kline = ('TWAP_min', 'count'),
    )


    bars = bars.sort_index(level = ['datetime', 'symbol'])
    bars['factor_name'] = f_name
    bars['factor_value'] = (
        bars
        .groupby('symbol')['TWAP_day']
        .rolling(window = roll_days)
        .mean()
        .reset_index(level = 0, drop = True)
        .rename('factor_value')
    )

    bars = bars.reset_index()
    return bars[['datetime', 'symbol', 'factor_name', 'factor_value']]


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
    factor = f_1228(bars, f_name="f_1228")
    print(f"因子计算完成:\n{factor.info()}\n{factor.tail()}")