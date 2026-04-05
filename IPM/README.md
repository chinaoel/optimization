# Convex QP Solvers & Portfolio Optimization

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![uv](https://img.shields.io/badge/Package_Manager-uv-purple.svg)
![NumPy](https://img.shields.io/badge/Math-NumPy-013243.svg)
![CVXPY](https://img.shields.io/badge/Benchmark-CVXPY-red.svg)

## Overview

This repository contains a from-scratch implementation of **Interior Point Methods (IPM)** for solving Convex Quadratic Programming (QP) problems. It was developed to deeply understand the mathematical machinery behind modern commercial solvers.

The project features two distinct algorithmic solvers and demonstrates their practical utility by solving the **Markowitz Portfolio Optimization** problem using real-world stock market data.

## Core Solvers (`/core`)

The solvers are built with a clean, object-oriented architecture (`BaseQPSolver` abstract base class) to ensure strict separation of concerns between data containers and algorithmic logic.

1. **Primal-Dual IPM (`primal_dual.py`)**:
   - Formulates and solves the full Karush-Kuhn-Tucker (KKT) system.
   - Features simultaneous primal and dual variable updates.
   - Implements a custom adaptive fraction-to-the-boundary step size and backtracking line search based on KKT residual norms.
   - Natively supports both inequality ($Gx \le h$) and equality ($Ax = b$) constraints.

2. **Log-Barrier Method (`log_barrier.py`)**:
   - A sequential unconstrained minimization approach.
   - Uses Newton's Method internally with a strictly feasible backtracking line search to handle inequality constraints via a logarithmic penalty function.

## Applications (`/applications`)

### 1. Markowitz Portfolio Optimization (`portfolio_opt.py`)

Fetches historical market data via `yfinance` to construct the Covariance matrix ($Q$) and Expected Returns ($c$). The problem is formulated to maximize returns while minimizing risk, strictly enforcing:

- **No short-selling** ($x \ge 0$)
- **Maximum position limits** ($x \le 30\%$ per asset to prevent corner solutions)
- **Fully invested budget** ($\sum x = 1$)

### 2. Efficient Frontier Generation (`efficient_frontier.py`)

Computes and visualizes the Constrained Efficient Frontier by systematically sweeping the risk-aversion parameter ($\lambda$) and utilizing **warm starts** for rapid IPM convergence.

## Performance Benchmark

The custom `PrimalDualSolver` is systematically benchmarked against **CVXPY** (using the highly-optimized C-based OSQP backend).

For small-to-medium dense matrices (e.g., a 6-asset portfolio), the custom pure-Python implementation exhibits **extreme efficiency**, converging to an $\epsilon < 10^{-8}$ tolerance in just ~11 iterations, often matching or outperforming the overhead-heavy DSL wrapper.

| Solver                   | Execution Time (sec) | Newton Iters |
| :----------------------- | :------------------- | :----------- |
| **CVXPY (OSQP Backend)** | ~ 0.0094             | N/A          |
| **Custom PRIMAL-DUAL**   | **~ 0.0011**         | **11**       |

_(Results generated via `uv run applications/portfolio_opt.py`)_

## Installation & Usage

This project utilizes [uv](https://github.com/astral-sh/uv) for lightning-fast, reproducible dependency management.

**1. Install `uv` (if not already installed):**

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```
