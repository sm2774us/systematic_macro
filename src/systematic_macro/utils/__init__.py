"""Utility functions: metrics, HLZ correction, covariance, Kelly, Hurst/Kalman, protocols."""

from systematic_macro.utils.covariance import CovarianceEstimator
from systematic_macro.utils.hlz import HLZCorrection, HLZResult, sharpe_to_tstat, tstat_to_sharpe
from systematic_macro.utils.hurst_kalman import HurstEstimator, KalmanFilter1D
from systematic_macro.utils.kelly import KellySizer
from systematic_macro.utils.metrics import (
    bonferroni_sharpe_threshold,
    compute_calmar,
    compute_ic,
    compute_icir,
    compute_marginal_sharpe,
    compute_max_drawdown,
    compute_net_ic,
    compute_rolling_ic,
    compute_sharpe,
)
from systematic_macro.utils.protocols import EquityETF, FuturesContract, FXSpot, TradeableAsset

__all__ = [
    "compute_ic", "compute_rolling_ic", "compute_icir", "compute_sharpe",
    "compute_max_drawdown", "compute_net_ic", "compute_marginal_sharpe",
    "compute_calmar", "bonferroni_sharpe_threshold",
    "HLZCorrection", "HLZResult", "sharpe_to_tstat", "tstat_to_sharpe",
    "CovarianceEstimator",
    "KellySizer",
    "HurstEstimator", "KalmanFilter1D",
    "TradeableAsset", "FXSpot", "FuturesContract", "EquityETF",
]
