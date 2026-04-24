# -*- coding: utf-8 -*-
"""
因子评估器模块

基于alphalens的数据格式和处理方式，提供因子有效性评估功能。
包含IC/ICIR分析、回归分析、分层回测、稳定性分析等功能。

Classes:
--------
FactorEvaluator : 因子评估器主类
    提供完整的因子评估分析功能

Main Features:
--------------
- 因子数据预处理（去极值、标准化）
- IC/ICIR分析（信息系数分析）
- 因子回归分析（Fama-MacBeth方法）
- 分层回测分析（多空组合表现）
- 稳定性分析（自相关性、换手率）
- 综合分析报告生成

Dependencies:
-------------
- pandas, numpy: 数据处理
- scipy.stats: 统计分析
- statsmodels: 回归分析
- empyrical: 风险收益指标计算
- alphalens: 因子分析核心库

Author: AI Assistant
Version: 2.0
Date: 2025-12
License: MIT
"""
import os
import sys
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, package_root)   # 新增python包检索路径

# 忽略警告信息
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import empyrical as ep
import alphalens as al
from joblib import Parallel, delayed


# ===================== 工具函数 =====================

def handle_outliers(
    series: pd.Series,
    method: str = "box",
    mad_k: float = 3.5,
    std_k: float = 3.0,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> pd.Series:
    """
    极值处理函数（单个截面用）

    Parameters
    ----------
    series : pd.Series
        单个截面的因子数据
    method : {"box", "std", "mad", "quantile", "none"}
        极值处理方式
        - "box":      箱线图法, 1.5 * IQR
        - "std":      标准差法, 均值 ± std_k * 标准差  
        - "mad":      中位数偏差法, 中位数 ± mad_k * MAD * 1.4826
        - "quantile": 分位数截断法, [q_low, q_high]
    std_k : float
        STD方法的倍数
    mad_k : float
        MAD方法的倍数
    q_low, q_high : float
        分位数法的上下界

    Returns
    -------
    pd.Series
        处理后的序列（索引不变）
    """

    if series is None or len(series) == 0:
        return series

    # 对于样本太少或全等的情况，直接返回
    if series.std() == 0 or len(series) < 5:
        return series

    method = (method or "box").lower()

    if method == "none":
        return series

    if method == "box":
        # 箱线图法, 1.5 * IQR
        q75 = series.quantile(0.75)
        q25 = series.quantile(0.25)
        iqr = q75 - q25
        if iqr == 0:
            return series
        upper = q75 + 1.5 * iqr
        lower = q25 - 1.5 * iqr
        return series.clip(lower, upper)

    elif method == "std":
        # 标准差法, 均值 ± std_k * 标准差  
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series
        upper = mean + std_k * std
        lower = mean - std_k * std
        return series.clip(lower, upper)

    elif method == "mad":
        # 中位数绝对偏差法，中位数 ± mad_k * MAD * 1.4826
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            # 没有波动就不动
            return series
        # 1.4826 是把MAD估计转换到和标准差同量纲的常数
        scale = mad * 1.4826
        upper = median + mad_k * scale
        lower = median - mad_k * scale
        return series.clip(lower, upper)

    elif method == "quantile":
        # 分位数截断法, [q_low, q_high]
        low_v = series.quantile(q_low)
        high_v = series.quantile(q_high)
        return series.clip(lower=low_v, upper=high_v)

    else:
        # 未知方式
        return series


def standardize(series: pd.Series) -> pd.Series:
    """
    标准化处理（截面内 Z-score）；
    对样本过少/方差为0/全NaN做兼容
    
    处理规则：
    - 全NaN或有效样本<2：保持NaN（无法标准化）
    - 标准差为0（所有值相同）：返回0（表示无离散度）
    - 正常情况：Z-score标准化
    """
    s = series.astype(float)
    # 有效样本数
    n = s.count()
    if n < 2:
        # 全NaN或样本不足，保持NaN
        return pd.Series(np.nan, index=s.index)
    std = s.std()
    if pd.isna(std) or std == 0:
        # 标准差为0，所有值相同，返回0
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def neutralize_factor(
    factor_data: pd.Series, 
    market_cap_data: pd.Series = None,
    industry_data: pd.Series = None,
    drop_first_industry: bool = True
) -> pd.Series:
    """
    因子中性化处理
    
    根据提供的市值和行业数据，对因子进行中性化处理：
    1. 仅市值数据：市值中性化
    2. 仅行业数据：行业中性化  
    3. 同时提供：市值+行业中性化
    
    Parameters:
    -----------
    factor_data : pd.Series
        因子数据，MultiIndex (datetime, symbol)
    market_cap_data : pd.Series, optional
        市值数据，MultiIndex (datetime, symbol)
        用户已经预处理好的数据（可能是原始市值、log(市值)或标准化后的值）
    industry_data : pd.Series, optional
        行业标签数据，MultiIndex (datetime, symbol)
        包含行业标签（如'Tech', 'Finance'等），将自动转换为哑变量
    drop_first_industry : bool, 默认 True
        生成行业哑变量时是否去掉第一个行业以避免共线性。
        
    Returns:
    --------
    neutralized_factor : pd.Series
        中性化后的因子数据
    """
    if market_cap_data is None and industry_data is None:
        return factor_data

    if not isinstance(factor_data.index, pd.MultiIndex) or factor_data.index.nlevels != 2:
        raise ValueError("factor_data 必须为 MultiIndex(datetime, symbol) 格式的 Series")
    
    def neutralize_cross_section(group):
        """对单个时间截面进行中性化处理"""
        # 共同索引, 对齐因子、市值、行业数据
        # 取因子、市值、行业数据三者的交集(旧版,已弃用)
        # common_index = group.index
        # if market_cap_data is not None:
        #     common_index = common_index.intersection(market_cap_data.index)
        # if industry_data is not None:
        #     common_index = common_index.intersection(industry_data.index)

        # 以因子索引为基准对齐市值、行业数据
        common_index = group.index
        
        # 获取当期因子值
        factor_values = group.loc[common_index].values
        
        # 构建回归变量
        X_vars = []
        
        # 添加市值变量
        if market_cap_data is not None:
            try:
                # 按对齐后的索引,获取当日市值数据
                date_market_cap = market_cap_data.reindex(common_index)
                market_cap_values = date_market_cap.values
                # 直接使用用户预处理好的市值数据，不做任何变换
                # 用户应在输入前根据需要进行取对数、标准化等处理
                X_vars.append(market_cap_values)
                
            except (KeyError, IndexError):
                # 如果该日期没有市值数据，跳过市值中性化
                pass
        
        # 添加行业哑变量
        if industry_data is not None:
            try:
                # 按对齐后的索引,获取当日行业标签数据（必须是Series格式）
                date_industry = industry_data.reindex(common_index)
                
                # 检查是否为Series且有多个行业
                if isinstance(date_industry, pd.Series):
                    unique_industries = date_industry.unique()
                    unique_industries = unique_industries[~pd.isna(unique_industries)]  # 去除NaN
                    
                    if len(unique_industries) > 1:
                        # 使用pd.get_dummies()将行业标签转换成哑变量
                        industry_dummies = pd.get_dummies(date_industry, drop_first=drop_first_industry)
                        
                        # 将哑变量添加到回归变量中
                        for col in industry_dummies.columns:
                            X_vars.append(industry_dummies[col].values)
                    
            except (KeyError, IndexError) as e:
                # 如果该日期没有行业数据，跳过行业中性化
                pass
        
        # 如果没有有效的回归变量，返回原始因子值
        if len(X_vars) == 0:
            return group
        
        # 构建设计矩阵
        X = np.column_stack(X_vars)
        
        # 标记有效样本(非缺失值)
        valid_mask = ~(np.isnan(factor_values) | np.any(np.isnan(X), axis=1))
        if valid_mask.sum() < max(10, len(X_vars) + 1):  # 数据点太少则跳过
            return group
        
        # 使用有效数据进行回归
        factor_valid = factor_values[valid_mask]
        X_valid = X[valid_mask]
        
        try:
            # 使用statsmodels进行线性回归
            # 注意：这里使用有截距项的模型，虽然理论上可以通过
            # 剔除截距项来包含所有行业哑变量，但在实际应用中：
            # 1. 有截距项的模型更稳健，能捕获整体市场水平
            # 2. 避免将整体水平错误地分配给各个行业
            # 3. 符合统计学和计量经济学的标准做法
            X_with_const = sm.add_constant(X_valid)  # 添加常数项
            model = sm.OLS(factor_valid, X_with_const).fit()
            
            # 计算残差（中性化后的因子值）
            X_all_with_const = sm.add_constant(X)
            predicted = model.predict(X_all_with_const)
            neutralized = factor_values - predicted
            
            # 使用有效数据的索引
            return pd.Series(neutralized, index=common_index)
            
        except Exception as e:
            # 回归失败时返回原始值
            date = group.index.get_level_values(0)[0]
            print(f"日期 {date} 中性化失败: {e}")
            return group
    
    # 按日期分组进行中性化
    neutralized_factor = factor_data.groupby(level=0, group_keys=False).apply(neutralize_cross_section)
    
    return neutralized_factor


def preprocess_factor_data(
    factor_data: pd.Series,
    market_cap_data: pd.Series = None,
    industry_data: pd.Series = None,
    drop_first_industry: bool = True,
    standardize_when: str = "both",
) -> pd.Series:
    """
    因子数据预处理：去极值、中性化和标准化
    
    处理顺序说明：
    1. 去极值：首先处理异常值，防止极端值影响后续处理
    2. 标准化：对原始因子进行标准化，确保符合正态分布
    3. 中性化：市值行业中性化，去除原始因子中的系统性偏差
    4. 标准化：最后对“纯净”因子进行标准化，确保可比性
    
    为什么中性化在标准化之前？
    - 先标准化再中性化可能会掩盖真实的市值/行业偏差
    - 标准化后的因子分布可能会放大某些异常值
    - 中性化在前能确保去除的是原始的、真实的系统性风险暴露
    
    Parameters:
    -----------
    factor_data : pd.Series
        原始因子数据，MultiIndex (datetime, symbol)
    market_cap_data : pd.Series, optional
        市值数据，MultiIndex (datetime, symbol)
        用户已经预处理好的数据（可以是原始市值、log(市值)或标准化后的值）
    industry_data : pd.Series, optional
        行业标签数据，MultiIndex (datetime, symbol)
        包含行业标签（如'Tech', 'Finance'等），将自动转换为哑变量
    drop_first_industry : bool, 默认 True
        生成行业哑变量时是否去掉第一个行业以避免共线性。
    standardize_when : str
        标准化阶段 ('pre'/'post'/'both'/'none')。
        
    Returns:
    --------
    processed_data : pd.Series
        处理后的因子数据
    """
    # 步骤1：极值处理 - 按时间截面处理异常值
    # 这确保了每个时间点的因子分布不会被极端值扭曲
    processed_data = factor_data.groupby(level=0, group_keys=False).apply(handle_outliers)

    # 步骤2：标准化处理 - 按时间截面进行Z-score标准化
    if standardize_when in {"pre", "both"}:
        processed_data = processed_data.groupby(level=0, group_keys=False).apply(standardize)
    
    # 步骤3：中性化处理（如果提供了市值或行业数据）
    processed_data = neutralize_factor(processed_data, market_cap_data, industry_data, drop_first_industry)
    
    # 步骤4：标准化处理 - 按时间截面进行Z-score标准化
    if standardize_when in {"post", "both"}:
        # 对中性化后的"纯净"因子进行标准化，确保不同时间点的因子值具有可比性
        processed_data = processed_data.groupby(level=0, group_keys=False).apply(standardize)
    
    return processed_data



# ===================== 功能函数 =====================

def get_clean_factor_and_forward_returns(
    factor_data: pd.Series,
    prices_data: pd.DataFrame, 
    market_cap_data: pd.Series = None, 
    industry_data: pd.Series = None,
    drop_first_industry: bool = True,
    standardize_when: str = "both",
    period=1, 
    quantiles=5,
    **kwargs
):
    """
    清洗因子数据并计算向前收益率
    
    参考alphalens.utils.get_clean_factor_and_forward_returns的实现，
    支持市值中性化、行业中性化或市值+行业中性化处理。
    
    数据预处理责任：
    - 评估器内部不对市值数据进行任何变换（如取对数、标准化）
    - 用户应在输入前根据具体需求对数据进行预处理
    - 评估器专注于核心的中性化回归逻辑
    
    Parameters:
    -----------
    factor_data : pd.Series
        因子数据，MultiIndex (datetime, symbol)
    prices_data : pd.DataFrame
        价格数据，index为datetime，columns为symbol
    market_cap_data : pd.Series, optional
        市值数据，MultiIndex (datetime, symbol)
        用户已经预处理好的数据（可以是原始市值、log(市值)或标准化后的值）
        提供时将进行市值中性化处理
    industry_data : pd.Series, optional
        行业数据：
        - 必须是Series格式：MultiIndex (datetime, symbol)，包含行业标签
        - 内容示例：'Tech', 'Finance', 'Healthcare', 'Energy'等
        - 将自动转换为哑变量进行回归
    drop_first_industry : bool, 默认 True
        生成行业哑变量时是否去掉第一个行业以避免共线性。
    standardize_when : str, 默认 "both"
        标准化阶段 ('pre'/'post'/'both'/'none')。
    period : int, default 1
        向前收益率计算周期（天数）
    quantiles : int, default 5
        分位数数量，用于因子分层
    **kwargs : dict
        传递给alphalens.utils.get_clean_factor_and_forward_returns的其他参数
        
    Returns:
    --------
    clean_data : pd.DataFrame
        包含因子值、因子分组、向前收益率的完整数据框
        index: MultiIndex (date, asset)
        columns: ['factor', 'factor_quantile', '{period}D']
    """

    # 数据预处理：去极值、中性化和标准化
    factor_data = preprocess_factor_data(
        factor_data, market_cap_data, industry_data, drop_first_industry, standardize_when
    )

    # 构造alphalens参数
    default_kwargs = {
        'factor': factor_data,
        'prices': prices_data,
        'periods': (period, ),
        'bins': None,
        'quantiles': quantiles,
        'groupby': None,
        'groupby_labels': None,
        'binning_by_group': False,
        'filter_zscore': 20,
        'max_loss': 0.35,
        'zero_aware': False,
        'cumulative_returns': True
    }
    default_kwargs.update(kwargs)
    
    # 调用alphalens进行数据清洗和前向收益率计算
    clean_data = al.utils.get_clean_factor_and_forward_returns(**default_kwargs)
    return clean_data


def ic_analysis(factor_data):
    """
    IC/ICIR分析，使用Spearman相关系数
    
    使用alphalens库的factor_information_coefficient方法计算信息系数（IC），
    并进行统计分析，包括IC均值、ICIR、胜率和显著性检验。
    
    Parameters:
    -----------
    factor_data : pd.DataFrame
        清洗后的因子数据，包含因子值、因子分组、向前收益率
        由get_clean_factor_and_forward_returns方法返回
    
    Returns:
    --------
    ic_ts : pd.Series
        IC时序数据
    ic_stats : dict
        IC统计指标，包含以下键值：
        - ic_mean: IC均值
        - icir_annualized: 年化信息比率（IC均值/IC标准差 * sqrt(annual_factor)）
        - ic_win_rate: IC胜率（根据IC方向调整）
        - ic_t_stat: t统计量
        - ic_p_value: p值
    """
    if factor_data is None or factor_data.empty:
        raise ValueError("factor_data不能为空")
    
    # 从factor_data中解析period
    period_cols = [col for col in factor_data.columns if col.endswith('D')]
    if not period_cols:
        raise ValueError("factor_data中未找到period列")
    period_col = period_cols[0]  # 取第一个收益率列
    period = int(period_col.replace('D', ''))

    # 使用alphalens计算信息系数(IC)时间序列
    # IC衡量因子值与未来收益率的线性相关性（基于Spearman相关系数）
    ic_data = al.performance.factor_information_coefficient(
        factor_data, 
        group_adjust=False,  # 不进行行业中性化调整
        by_group=False       # 不按组别分别计算
    )
    
    # IC时序数据
    ic_ts = ic_data[period_col]
    if ic_ts.empty or len(ic_ts) == 0:
        return None, None
    
    # 计算IC
    ic_mean = ic_ts.mean()
    ic_std = ic_ts.std()
    # 计算年化ICIR（信息系数比率）
    icir = ic_mean / ic_std if ic_std != 0 else 0.0
    annual_factor = pd.Timedelta("242D") / pd.Timedelta(period_col)
    icir_annualized = icir * np.sqrt(annual_factor)

    # 计算IC胜率（考虑IC方向）
    if ic_mean > 0:
        ic_win_rate = (ic_ts>0).mean()
    else:
        ic_win_rate = (ic_ts<0).mean()
    
    # 显著性检验
    t_stat, p_value = stats.ttest_1samp(ic_ts, 0)
    
    # 汇总评估结果
    ic_stats = {
        'ic_mean': round(ic_mean, 4),
        'icir_annualized': round(icir_annualized, 4),
        'ic_win_rate': round(ic_win_rate, 4),
        'ic_t_stat': round(t_stat, 4),
        'ic_p_value': round(p_value, 4),
    }
    
    return ic_ts, ic_stats


def regression_analysis(factor_data):
    """
    因子回归分析：使用截面回归方法分析因子收益率
    
    基于Fama-MacBeth方法，在每个时间截面对所有股票的因子值和向前收益率进行回归，
    得到因子收益率时间序列，然后分析因子的统计特性。
    
    Parameters:
    -----------
    factor_data : pd.DataFrame
        清洗后的因子数据，包含因子值、因子分组、向前收益率
        由get_clean_factor_and_forward_returns方法返回
    
    Returns:
    --------
    tuple : (factor_returns_ts, regression_stats)
        - factor_returns_ts: pd.Series, 因子收益率时间序列
        - regression_stats: dict, 回归分析统计结果
            - factor_return: 因子平均收益率
            - ols_t_stat: 整体T统计量（基于因子收益率时间序列）
            - ols_|t_stat|>2: T统计量绝对值>2的比率（稳定性指标）
            - ols_r_squared: 平均R²值（因子解释能力）
            - ols_hit_rate: 胜率（根据因子方向性调整的成功率）
    """
    if factor_data is None or factor_data.empty:
        raise ValueError("factor_data不能为空")

    def _cross_sectional_regression(group):
        """
        截面回归函数
        
        对单个时间截面的因子值和收益率进行线性回归，
        计算因子收益率、t统计量和R²值。
        
        Parameters:
        -----------
        group : pd.DataFrame
            单个时间点的因子和收益率数据
            
        Returns:
        --------
        pd.DataFrame
            包含以下列的DataFrame：
            - datetime: 日期
            - factor_rets: 因子收益率（回归系数）
            - t-stat: t统计量
            - r_squared: R²值
        """
        # 获取日期
        date = group.index.get_level_values(0)[0]
            
        # 去除缺失值
        valid_data = group.dropna()
        if len(valid_data) < 10:
            return pd.DataFrame({
                'datetime': [date],
                'factor_rets': [np.nan],
                't-stat': [np.nan],
                'r_squared': [np.nan]
            })
    
        try:
            # 从factor_data中解析period
            period_cols = [col for col in valid_data.columns if col.endswith('D')]
            period_col = period_cols[0]
            
            # 提取因子值和收益率
            X = valid_data['factor']
            y = valid_data[period_col]
            
            # 添加常数项
            X_with_const = sm.add_constant(X)

            # 进行线性回归
            model = sm.OLS(y, X_with_const).fit()
            
            # 获取因子系数（斜率）作为因子收益率
            factor_return = model.params['factor']
            t_stat = model.tvalues['factor']
            r_squared = model.rsquared
            
            return pd.DataFrame({
                'datetime': [date],
                'factor_rets': [factor_return],
                't-stat': [t_stat],
                'r_squared': [r_squared]
            })
            
        except Exception as e:
            # 回归失败时返回NaN
            return pd.DataFrame({
                'datetime': [date],
                'factor_rets': [np.nan],
                't-stat': [np.nan],
                'r_squared': [np.nan]
            })
    
    # 按日期分组并应用截面回归函数
    ols_results = factor_data.groupby(level=0, group_keys=False).apply(_cross_sectional_regression)
    if ols_results.dropna().empty:
        raise ValueError("无法计算因子收益率，请检查数据质量")
        
    # 构建因子收益率时间序列
    factor_returns_ts = pd.Series(
        data=ols_results['factor_rets'].values,
        index=pd.DatetimeIndex(ols_results['datetime']),
        name='factor_returns'
    )
    
    # 计算统计指标
    mean_return = ols_results['factor_rets'].mean()
    std_return = ols_results['factor_rets'].std()
    
    # 胜率计算：根据因子方向性调整
    # 如果平均收益率为正（正向因子），胜率为>0的比例
    # 如果平均收益率为负（反向因子），胜率为<0的比例
    if mean_return >= 0:
        hit_rate = (ols_results['factor_rets'] > 0).mean()
    else:
        hit_rate = (ols_results['factor_rets'] < 0).mean()

    # 计算T统计量 -- 衡量因子有效性
    t_stat_overall = mean_return / (std_return / np.sqrt(len(ols_results))) if std_return > 0 else 0
    t_stat_gt_2_ratio = len(ols_results[ols_results['t-stat'].abs() > 2]) / len(ols_results)
    
    # 计算R-squared -- 衡量因子可解释性
    r_squared_avg = ols_results['r_squared'].mean()
    
    # 返回回归分析结果
    regression_stats = {
        'factor_return': mean_return,
        'ols_t_stat': t_stat_overall,
        'ols_|t_stat|>2': t_stat_gt_2_ratio,
        'ols_r_squared': r_squared_avg,
        'ols_hit_rate': hit_rate,
    }
    
    return factor_returns_ts, regression_stats


def layered_backtest(factor_data):
    """
    分层回测分析：将股票按因子值分组，分析各组收益率表现
    
    利用alphalens.performance模块的函数进行分层回测分析，
    计算各分层收益率、单调性和多空组合表现。
    
    Parameters:
    -----------
    factor_data : pd.DataFrame
        清洗后的因子数据，包含因子值、因子分组、向前收益率
        由get_clean_factor_and_forward_returns方法返回
    
    Returns:
    --------
    quantile_returns : pd.DataFrame
        各分层收益率时间序列，index为日期，columns为分层编号
    backtest_stats : dict
        分层回测统计结果，包含以下键值：
        - monotonicity: 单调性（分层与收益率的相关性）
        - top_annual_return: top层年化收益率
        - top_annual_volatility: top层年化波动率
        - top_max_drawdown: top层最大回撤
        - top_sharpe_ratio: top层夏普比率
        - top_calmar_ratio: top层卡玛比率
        - ls_annual_return: 多空组合年化收益率
        - ls_annual_volatility: 多空组合年化波动率
        - ls_max_drawdown: 多空组合最大回撤
        - ls_sharpe_ratio: 多空组合夏普比率
        - ls_calmar_ratio: 多空组合卡玛比率

        TODO TOP组 vs 基准对比
    """
    if factor_data is None or factor_data.empty:
        raise ValueError("factor_data不能为空")
    
    # 从factor_data中解析period
    period_cols = [col for col in factor_data.columns if col.endswith('D')]
    if not period_cols:
        raise ValueError("factor_data中未找到forward returns")
    period_col = period_cols[0]
    period = int(period_col.replace('D', ''))
    
    # 使用alphalens的mean_return_by_quantile函数计算各分层收益率
    # 计算各分层在多个period下的平均收益率时序DataFrame
    mean_ret_quant_bydate, std_quant_bydate = al.performance.mean_return_by_quantile(
        factor_data=factor_data, 
        by_date=True, 
        by_group=False, 
        demeaned=False, 
        group_adjust=False
    )
    # 将收益率转换为指定基准期间的收益率 eg. 20D -> 1D
    mean_ret_quant_bydate = mean_ret_quant_bydate.apply(
        al.utils.rate_of_return,
        axis=0,
        base_period="1D",
    )
    
    # 构建分层收益率时间序列
    # mean_ret_quant_bydate的索引是MultiIndex: (factor_quantile, date)
    # 需要重新索引以便正确unstack
    quantile_data = mean_ret_quant_bydate[period_col].reset_index()
    quantile_returns = quantile_data.pivot(index='date', columns='factor_quantile', values=period_col)
    quantile_returns.columns = [int(col) for col in quantile_returns.columns]
    
    # 计算各分层的年化收益率
    # 年化因子 = 242个交易日 / 持有期间
    annualization_factor = 242 / period
    quantile_annual_returns = {}
    for col in quantile_returns.columns:
        returns_series = quantile_returns[col].dropna()
        if len(returns_series) > 0:
            # 使用empyrical计算年化收益率
            annual_return = ep.annual_return(returns_series, annualization=annualization_factor)
            quantile_annual_returns[col] = annual_return
        else:
            quantile_annual_returns[col] = np.nan
    
    # 计算单调性：分层序号与年化收益率的Spearman相关性
    quantile_nums = [int(col) for col in quantile_annual_returns.keys()]
    annual_returns_values = list(quantile_annual_returns.values())
    if len(quantile_nums) >= 3 and not any(np.isnan(annual_returns_values)):
        monotonicity = abs(stats.spearmanr(quantile_nums, annual_returns_values)[0])
    else:
        monotonicity = np.nan

    # 划分top组和bottom组
    min_num, max_num = min(quantile_nums), max(quantile_nums)
    if quantile_annual_returns[min_num] > quantile_annual_returns[max_num]:
        top_quantile = min_num
        bottom_quantile = max_num
    else:
        top_quantile = max_num
        bottom_quantile = min_num
    # print(f"top组：{top_quantile}, bottom组：{bottom_quantile}")
    top_returns = quantile_returns[top_quantile].dropna()
    bottom_returns = quantile_returns[bottom_quantile].dropna()

    # 计算top层的各项收益指标
    if len(top_returns) > 0:
        top_cum_return = ep.cum_returns_final(top_returns, starting_value=0)
        top_annual_return = ep.annual_return(top_returns, annualization=annualization_factor)
        top_annual_volatility = ep.annual_volatility(top_returns, annualization=annualization_factor)
        top_max_drawdown = ep.max_drawdown(top_returns)
        top_sharpe_ratio = ep.sharpe_ratio(top_returns, annualization=annualization_factor)
        top_calmar_ratio = ep.calmar_ratio(top_returns, annualization=annualization_factor)
    else:
        top_cum_return = np.nan
        top_annual_return = np.nan
        top_annual_volatility = np.nan
        top_max_drawdown = np.nan
        top_sharpe_ratio = np.nan
        top_calmar_ratio = np.nan
    
    # 使用alphalens的compute_mean_returns_spread计算多空组合
    mean_ret_spread_quant, std_spread_quant = al.performance.compute_mean_returns_spread(
        mean_returns=mean_ret_quant_bydate,
        upper_quant=top_quantile,
        lower_quant=bottom_quantile,
        std_err=std_quant_bydate
    )
    
    # 提取对应周期的多空组合收益率
    long_short_returns = mean_ret_spread_quant[period_col].dropna()
    if len(long_short_returns) > 0:
        # 使用empyrical计算多空组合的各项指标
        ls_annual_return = ep.annual_return(long_short_returns, annualization=annualization_factor)
        ls_annual_volatility = ep.annual_volatility(long_short_returns, annualization=annualization_factor)
        ls_max_drawdown = ep.max_drawdown(long_short_returns)
        ls_sharpe_ratio = ep.sharpe_ratio(long_short_returns, annualization=annualization_factor)
        ls_calmar_ratio = ep.calmar_ratio(long_short_returns, annualization=annualization_factor)
    else:
        ls_annual_return = np.nan
        ls_annual_volatility = np.nan
        ls_max_drawdown = np.nan
        ls_sharpe_ratio = np.nan
        ls_calmar_ratio = np.nan

    # 构建回测数据
    backtest_data = {
        'quantile_returns': quantile_returns,
        'top_returns': top_returns,
        'bottom_returns': bottom_returns,
        'ls_returns': long_short_returns
    }
    
    # 构建回测结果
    backtest_stats = {
        'monotonicity': monotonicity,
        'top_cum_return': top_cum_return,
        'top_annual_return': top_annual_return,
        'top_annual_volatility': top_annual_volatility,
        'top_max_drawdown': top_max_drawdown,
        'top_sharpe_ratio': top_sharpe_ratio,
        'top_calmar_ratio': top_calmar_ratio,
        'ls_annual_return': ls_annual_return,
        'ls_annual_volatility': ls_annual_volatility,
        'ls_max_drawdown': ls_max_drawdown,
        'ls_sharpe_ratio': ls_sharpe_ratio,
        'ls_calmar_ratio': ls_calmar_ratio,
    }
    
    return backtest_data, backtest_stats


def stability_analysis(factor_data):
    """
    稳定性分析
    
    分析因子的稳定性，包括：
    1. 因子自相关性：衡量因子排名的时间稳定性
    2. 分位数换手率：衡量各分位数组合的换手频率
    
    Parameters:
    -----------
    factor_data : pd.DataFrame
            清洗后的因子数据，包含因子值、因子分组、向前收益率
            由get_clean_factor_and_forward_returns方法返回
    
    Returns:
    --------
    stability_stats : dict
        稳定性统计结果，包含以下键值：
        - factor_autocorr: 因子自相关性均值
        - overall_turnover: 所有分位数平均换手率
        - top_bottom_turnover: 顶层和底层平均换手率
    """
    if factor_data is None or factor_data.empty:
        raise ValueError("factor_data不能为空")
    
    # 从factor_data中解析period
    period_cols = [col for col in factor_data.columns if col.endswith('D')]
    if not period_cols:
        raise ValueError("factor_data中未找到收益率列")
    period_col = period_cols[0]
    period = int(period_col.replace('D', ''))
    
    # 1. 因子自相关性分析
    factor_autocorr = al.performance.factor_rank_autocorrelation(
        factor_data, period=period
    )
    # 计算自相关性的平均值作为稳定性指标
    autocorr_mean = factor_autocorr.mean()
    
    # 2. 分组换手率分析
    # 换手率衡量组合成分变化的频率，影响交易成本
    quantile_factor = factor_data['factor_quantile']
    unique_quantiles = quantile_factor.sort_values().unique()
    quantile_turnover = pd.concat([
        al.performance.quantile_turnover(quantile_factor, q, period)
        for q in unique_quantiles], axis=1
    )

    # 顶层+底层平均换手率
    top_quantile, bottom_quantile = max(unique_quantiles), min(unique_quantiles)
    top_bottom_turnover = quantile_turnover[[top_quantile, bottom_quantile]].mean(axis=1).mean()  
    # 所有分位数平均换手率
    overall_turnover = quantile_turnover.mean(axis=1).mean()
    
    # 构建返回结果
    stability_stats = {
        'factor_autocorr': autocorr_mean,
        'overall_turnover': overall_turnover,
        'top_bottom_turnover': top_bottom_turnover,
    }
    
    return stability_stats



# ===================== 工厂函数 =====================

def evaluate_factor(
        factor_data, prices_data, 
        market_cap_data=None, industry_data=None,
        period=1, quantiles=5, **kwargs
    ):
    """
    因子评估工厂函数 - 一站式因子评估
    
    提供完整的因子评估流程，包括数据清洗、IC分析、回归分析、
    分层回测和稳定性分析，返回综合评估报告。
    
    Parameters:
    -----------
    factor_data : pd.Series
        因子数据，MultiIndex (datetime, symbol)
    prices_data : pd.DataFrame
        价格数据，index为datetime，columns为symbol
    market_cap_data : pd.Series, optional
        市值数据，用于市值中性化
    industry_data : pd.Series, optional
        行业数据，用于行业中性化
    period : int, default 1
        前向收益率周期
    quantiles : int, default 5
        分层数量
    **kwargs : dict
        其他传递给alphalens的参数
        
    Returns:
    --------
    report : pd.Series
        综合分析指标的Series，包含所有分析模块的统计结果：
        - IC分析结果（ic_mean, icir_annualized, ic_win_rate等）
        - 回归分析结果（factor_return, ols_t_stat等）
        - 分层回测结果（monotonicity, top_annual_return等）
        - 稳定性分析结果（factor_autocorr, overall_turnover等）
    
    Examples:
    ---------
    >>> # 基础评估
    >>> result = evaluate_factor(factor_data, prices_data)
    >>> print(result)
    
    >>> # 带中性化的评估
    >>> result = evaluate_factor(
    ...     factor_data, prices_data, 
    ...     market_cap_data=market_cap, 
    ...     period=5, quantiles=10
    ... )
    """
    # 忽略警告信息
    import warnings
    warnings.filterwarnings(action='ignore')

    # 步骤1：数据清洗和前向收益率计算
    clean_data = get_clean_factor_and_forward_returns(
        factor_data, prices_data, 
        market_cap_data=market_cap_data,
        industry_data=industry_data,
        period=period, quantiles=quantiles,
        **kwargs
    )

    if clean_data.empty:
        raise ValueError("数据清洗后为空，请检查输入数据")
    
    # 步骤2：多角度进行因子评估
    ic_ts, ic_stats = ic_analysis(clean_data)
    factor_returns_ts, regression_stats = regression_analysis(clean_data)
    backtest_data, backtest_stats = layered_backtest(clean_data)
    stability_stats = stability_analysis(clean_data)
    
    # 步骤3：生成综合报告
    comprehensive_report = {}
    # 整合IC分析结果：因子预测能力指标
    if ic_stats:
        comprehensive_report.update(ic_stats)
    
    # 整合回归分析结果：因子收益率和统计显著性指标
    if regression_stats:
        comprehensive_report.update(regression_stats)
    
    # 整合分层回测结果：分层效果和多空策略表现指标
    if backtest_stats:
        comprehensive_report.update(backtest_stats)
    
    # 整合稳定性分析结果：因子稳定性和交易成本相关指标
    if stability_stats:
        comprehensive_report.update(stability_stats)

    # 添加因子名称
    if comprehensive_report:
        comprehensive_report['factor_name'] = factor_data.name
    
    return pd.Series(comprehensive_report)


def batch_evaluate_factors(
        factors_data, prices_data,
        market_cap_data=None, industry_data=None,
        period=1, quantiles=5, n_jobs=1,
        save_report=False, save_dir="./", filename_prefix="factor_report",
        **kwargs
    ):
    """
    批量因子评估工厂函数
    
    对多个因子进行批量评估，生成综合评估报告。支持多进程并行处理。
    
    Parameters:
    -----------
    factors_data : pd.DataFrame
        因子数据框，MultiIndex (datetime, symbol) x factors
    prices_data : pd.DataFrame
        价格数据，index为datetime，columns为symbol
    market_cap_data : pd.Series, optional
        市值数据，用于市值中性化
    industry_data : pd.Series, optional
        行业数据，用于行业中性化
    period : int, default 1
        前向收益率周期
    quantiles : int, default 5
        分层数量
    save_report : bool, default False
        是否保存报告到CSV文件
    save_dir: str, default "./"
        csv文件保持目录
    filename_prefix : str, default "factor_report"
        保存文件的前缀名
    n_jobs : int, default 1
        并行处理的进程数
        - 1: 单进程处理（默认）
        - -1: 使用所有可用CPU核心
        - >1: 使用指定数量的进程
    **kwargs : dict
        其他传递给因子评估的参数
        
    Returns:
    --------
    pd.DataFrame : 所有因子的综合评估报告
    """
    # print("开始批量因子评估...")
    reports = []
    
    # 确定要评估的因子列表
    factor_names = factors_data.columns
    
    # 根据n_jobs参数选择处理方式
    if n_jobs == 1:
        # 单进程处理
        for i, factor_name in enumerate(factor_names, 1):
            print(f"\n[{i}/{len(factor_names)}] 正在处理因子: {factor_name}")
            
            # 提取单个因子数据
            factor_data = factors_data[factor_name]
            
            # 调用单因子评估函数
            result = evaluate_factor(
                factor_data, prices_data,
                market_cap_data=market_cap_data,
                industry_data=industry_data,
                period=period, quantiles=quantiles,
                **kwargs
            )
            
            if not result.empty:
                report = result.copy()
                reports.append(report)
                print(f"  - 因子 {factor_name} 评估完成")
            else:
                print(f"  - 因子 {factor_name} 评估失败")

    else:
        # 多进程并行处理
        # print(f"使用 {n_jobs if n_jobs > 0 else '所有可用'} 个进程进行并行处理...")
        
        # 并行执行
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_factor)(
                factors_data[factor_name], prices_data,
                market_cap_data=market_cap_data,
                industry_data=industry_data,
                period=period, quantiles=quantiles,
                **kwargs
            ) for factor_name in factor_names
        )
        reports = results
        # print(f"共 {len(results)} 个因子评估完成")
    
    # 生成综合报告
    if reports:
        report_df = pd.DataFrame(reports)
        
        if save_report:
            # 构建文件名
            neutralization_info = []
            if market_cap_data is not None:
                neutralization_info.append("market_cap")
            if industry_data is not None:
                neutralization_info.append("industry")
            
            filename = filename_prefix
            if neutralization_info:
                filename += f"_neutralized_{'_'.join(neutralization_info)}"
            filename += ".csv"
            
            file_path = rf"{save_dir}/{filename}"
            report_df.to_csv(file_path, index=False)
            print(f"\n批量评估完成，报告已保存至: {file_path}")
        
        return report_df
    else:
        print("\n没有生成任何评估报告")
        return pd.DataFrame()


if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    import polars as pl

    try:
        import config_local as config
    except ImportError:
        import config

    # 导入数据接口sdk
    import zenidatasdk as zd
    client = zd.Client(
        base_url=config.ZENI_URL,
        username=config.ZENI_USERNAME,
        password=config.ZENI_PASSWORD, 
    )

    # 历史数据获取区间
    init_date = '2024-01-01'
    start_date = '2025-01-01'
    end_date = '2025-06-30'
    index_symbol = "000905.XSHG"
    factor_names = ["size", "beta", "momentum", "residual_volatility", "non_linear_size", "book_to_price_ratio", "liquidity", "earnings_yield", "growth", "leverage"]

    # 获取指数成分股数据
    print("正在获取指数成分股数据...")
    index_weights_df = client.get_index_constituents_df(
        index_symbol=index_symbol,
        start_date=start_date,
        end_date=end_date
    )
    index_weights_df = index_weights_df.rename(columns={"date": "datetime"})
    symbols = index_weights_df["symbol"].unique().tolist()

    # 获取日频bar数据
    print("正在获取日频bar数据...")
    bars_1d_df = client.kline.get_kline_df(
        symbol=symbols,
        start_date=init_date,
        end_date=end_date,
        frequency="1d",
        adjust_type="pre",
        market="cn_stock"
    )

    # 多资产价格数据(开盘价买入)
    prices_df = bars_1d_df[bars_1d_df["datetime"] >= start_date]
    prices = prices_df.pivot(index="datetime", columns="symbol", values="open")
    print(f"价格数据形状: {prices.shape}")
    print("价格数据预览:")
    print(prices.tail())

    # 获取zenidata中的因子数据
    print("正在获取因子数据...")
    factors_df = client.get_factors_df(
        symbols=symbols,
        factor_names=factor_names,
        start_date=start_date,
        end_date=end_date,
        market="cn_stock"
    )
    # 因子值 shift 1 转换成实际使用时间(T+1)
    factors_df["factor_value"] = factors_df.groupby('symbol')['factor_value'].transform(lambda x: x.shift(1))
    # 与指数的交易日历、历史成分股数据对齐
    factors_df = pd.merge(index_weights_df[["datetime", "symbol"]], factors_df, how="left", on=["datetime", "symbol"])
    # 转换成带[datetime, symbol]双重索引的factor_table
    factors = factors_df.pivot_table(index=["datetime", "symbol"], columns="factor_name", values="factor_value")
    print(f"因子数据形状: {factors.shape}")
    print(f"包含因子: {list(factors.columns)}")
    print("因子数据预览:")
    print(factors.tail())

    # ==================== 使用新的函数式接口 ====================
    
    # 获取市值数据（用于市值中性化）
    market_cap = None
    industry = None
    
    if 'size' in factors.columns:
        market_cap = factors['size']
        print("检测到市值数据，将进行市值中性化处理")
    
    # 方法1：单因子评估示例
    print("\n=== 单因子评估示例 ===")
    if 'momentum' in factors.columns:
        result = evaluate_factor(
            factors['momentum'], 
            prices,
            market_cap_data=market_cap,
            period=1, 
            quantiles=5
        )

        print("动量因子评估报告:")
        print(result)
    
    # 方法2：批量因子评估（推荐）
    print("\n=== 批量因子评估 ===")
    report_df = batch_evaluate_factors(
        factors, 
        prices,
        market_cap_data=market_cap,
        industry_data=industry,
        period=1,
        quantiles=5,
        n_jobs=8,
        save_report=False,
        filename_prefix="batch_factor_report"
    )
    
    if not report_df.empty:
        print("\n批量评估结果预览:")
        print(report_df[['factor_name', 'ic_mean', 'icir_annualized', 'top_annual_return', 'ls_annual_return']].head())
