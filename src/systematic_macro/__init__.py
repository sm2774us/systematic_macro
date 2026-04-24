# Copyright 2026 Systematic Macro Research. All rights reserved.
#
# Licensed under the MIT License. See LICENSE for details.
#
# Systematic Macro Signal Research Pipeline
# Carry · Momentum · Flow  —  FX · Futures · Equities
# Designed for 2026 macro/geopolitical environment.
"""Systematic Macro: end-to-end signal research for FX, Futures, and Equities.

Modules:
    data:       Market data fetching and preprocessing.
    signals:    Carry, momentum, and flow/positioning signals.
    portfolio:  Mean-variance and risk-parity portfolio optimisation.
    backtest:   Walk-forward backtesting engine with net-IC gating.
    utils:      Statistical metrics (ICIR, Sharpe, MDD, TCA).
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("systematic-macro")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
