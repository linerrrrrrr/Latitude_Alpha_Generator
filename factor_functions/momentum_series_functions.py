import alphalens
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import os

# ================  生成 zylh 规范的因子计算函数  ==================

def f_momentum20D_001(bars: pd.DataFrame, f_name = 'f_0001', min_periods: int = 20) -> pd.DataFrame:
    """
    factor_intro: 1个月动量因子，20日收益率
    category: momentum
    category_intro: 动量类因子
    subcategory: short_term_momentum
    subcategory_intro: 短期动量
    min_periods: 20
    source: None
    author: 魏丰协

    Parameters
    ----------
    bars: pd.DataFrame
        日频, bars
        ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    f_name: str
        因子名称
    min_periods: int
        计算窗口期，默认为20（1个月交易日）

    Returns
    -------
    pd.DataFrame
        因子值，包含列：['datetime', 'symbol', 'factor_name', 'factor_value']
    """
    close_2dim = bars.reset_index()[                                        # 透视为二维交叉表
        ['datetime', 'symbol', 'close']
    ].pivot(
        index = 'datetime',
        columns = 'symbol',
        values = 'close'
    )
    close_2dim = close_2dim.ffill()
    momentum_2dim = close_2dim.pct_change(periods = min_periods).shift(1)   # 计算变动率并在时间轴上前进一格
    momentum_long = momentum_2dim.reset_index().melt(                       # 融列为长表
        id_vars = ['datetime'],
        var_name = 'symbol',
        value_name = 'factor_value'
    ).dropna()

    momentum_long.columns = ['datetime', 'symbol', 'factor_value']          # 对列名重命名
    momentum_long['factor_name'] = rf"momentum_{min_periods}"               # 添加因子名称列以方便库管理

    momentum_long = momentum_long[['datetime', 'symbol', 'factor_name', 'factor_value']].copy()
    return momentum_long

factor_for_zylh = f_momentum20D_001(bars = big_df, f_name = 'f_0001', min_periods = 20) # 智盈量化因子入库格式规范
factor_for_zylh.head()
