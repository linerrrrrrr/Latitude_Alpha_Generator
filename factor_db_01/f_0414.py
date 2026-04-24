import numpy as np
import pandas as pd


def f_db_0414(bars: pd.DataFrame, f_name: str = 'f_db_0414', roll_days: int = 20) -> pd.DataFrame:
    """
    factor_intro: 上下影线/收盘价的标准差
    category: 日频因子
    category_intro: 量价因子
    subcategory:
    subcategory_intro: 量价因子改进
    min_period: 20d
    source: 王琦, 2023, JASON's alpha: 基本面+量价复合策略, 东北证券
    author: 因子团队

    说明：使用每日K线的(上影线长度+下影线长度)/当日收盘价，并在月末计算当月的数据标准差，
    作为原始因子值。根据以上计算过程，判断因子方向为负向。
    因子逻辑：上下影线表现了多空博弈状态，通过选择上下影线总长度较短的标的，规避博弈剧烈、稳定性较差的股票

    Parameters
    ----------
    bars: pd.DataFrame
        日频 bars
        ['datetime', 'symbol', 'open', 'high', 'low', 'close']
    f_name: str
        因子名称，默认为 'f_db_0414'
    roll_days: int
        滚动窗口天数，默认为 20。研报原文为每月末计算当月标准差，
        本实现采用日频过去 roll_days 日滚动标准差替代。

    Returns
    -------
    factors: pd.DataFrame
        日频 factors
        ['datetime', 'symbol', 'factor_name', 'factor_value']
    """



    # 用语规范：
    # 柱体实体；Real Body
    # 上影线；Upper Shadow（规范）、Upper Wick（常用）
    # 下影线；Lower Shadow（规范）、Lower Wick（常用）
    # 柱体上沿：Top of the Body
    # 柱体的下沿：Bottom of the Body
    # 注：wick n. 灯芯、蜡烛芯


    # 定义阳线 与 非阳线（即阴线或十字星）
    # 再根据阳线布尔值，确定蜡烛柱体上下沿
    bars['is_rising'] = bars['open'] < bars['close']
    bars['body_top'] = np.where(bars['is_rising'], bars['close'], bars['open'])
    bars['body_bottom'] = np.where(bars['is_rising'], bars['open'], bars['close'])

    # 计算上下影线长度
    bars['upper_shadow'] = bars['high'] - bars['body_top']
    bars['lower_shadow'] = bars['body_bottom'] - bars['low']

    # 计算上下影线长度之和与收盘价的比值 Std of shadow to close
    bars = bars.set_index(['datetime', 'symbol'])
    bars['factor_value'] = (bars['upper_shadow'] + bars['lower_shadow'])/bars['close']
    roll_stats = bars.groupby('symbol')['factor_value'].rolling(window = roll_days).agg('std').reset_index(level = 0, drop = True)

    bars = pd.merge(left = bars[[]], right = roll_stats, on = ['datetime', 'symbol'], how = 'left')
    bars['factor_value'] *= -1
    bars['factor_name'] = f_name

    return bars


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

    # print("获取分钟K线数据...")
    # bars = client.get_kline_df(
    #     symbol=symbols,
    #     start_date=init_date,
    #     end_date=end_date,
    #     frequency="1m",
    #     adjust_type="post",
    #     market="cn_stock",
    # )
    print("获取日频K线数据...")
    bars = client.kline.get_kline_df(
        symbol=symbols,
        start_date=init_date,
        end_date=end_date,
        frequency="1d",
        adjust_type="post",
        market="cn_stock"
    )

    print("计算因子...")
    factor = f_db_0414(bars, f_name="f_0107")
    print(f"因子计算完成:\n{factor.info()}\n{factor.tail()}")