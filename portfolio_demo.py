import os
import sys
import time
import argparse
import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

# 確保 Python 能找到外層的 core 模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from IPM.core.qp_problem import QPProblem
from IPM.core.primal_dual import PrimalDualSolver
from IPM.core.log_barrier import LogBarrierSolver

def get_market_data(tickers: list, start_date: str, end_date: str):
    """從 Yahoo Finance 抓取歷史價格並計算年化報酬與共變異數"""
    print(f"Fetching data for: {', '.join(tickers)}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data[tickers]
    
    returns = data.pct_change().dropna()
    trading_days = 252
    
    annual_returns = returns.mean() * trading_days
    annual_cov = returns.cov() * trading_days
    
    return annual_returns.values, annual_cov.values

def main():
    # ==========================================
    # 1. 解析命令列參數 (CLI Arguments)
    # ==========================================
    parser = argparse.ArgumentParser(description="Portfolio Optimization using Custom IPM Solvers.")
    parser.add_argument(
        '--solver', 
        type=str, 
        choices=['primal_dual', 'log_barrier'], 
        default='primal_dual',
        help="Choose the custom solver to run against CVXPY (default: primal_dual)"
    )
    args = parser.parse_args()
    
    print(f"=== Starting Portfolio Optimization Demo ===")
    print(f"Selected Custom Solver: {args.solver.upper()}\n")

    # ==========================================
    # 2. 定義資產池與抓取數據
    # ==========================================
    tickers = ['AAPL', 'NVDA', 'JNJ', 'PG', 'TLT', 'GLD']
    expected_returns, cov_matrix = get_market_data(tickers, "2020-01-01", "2024-01-01")

    n = len(tickers)
    Q = cov_matrix
    risk_aversion = 1.0
    c = -risk_aversion * expected_returns

    # 持股上限設定 (避免過度集中在單一飆股)
    max_weight = 0.30

    # ==========================================
    # 3. 執行 CVXPY (Baseline Benchmark)
    # ==========================================
    print("\nRunning CVXPY (Commercial Baseline)...")
    x_cvx = cp.Variable(n)
    objective_cvx = cp.Minimize((1/2) * cp.quad_form(x_cvx, Q) + c.T @ x_cvx)
    
    # 配合所選 Solver 決定 CVXPY 的限制式，統一加入持股上限
    if args.solver == 'primal_dual':
        constraints_cvx = [x_cvx >= 0, x_cvx <= max_weight, cp.sum(x_cvx) == 1.0] 
    else:
        constraints_cvx = [x_cvx >= 0, x_cvx <= max_weight, cp.sum(x_cvx) <= 1.0] 
        
    prob_cvx = cp.Problem(objective_cvx, constraints_cvx)
    
    t0_cvx = time.perf_counter()
    prob_cvx.solve()
    t1_cvx = time.perf_counter()
    time_cvxpy = t1_cvx - t0_cvx
    
    cvxpy_weights = x_cvx.value

    # ==========================================
    # 4. 執行 Custom Solver
    # ==========================================
    print(f"Running Custom {args.solver.upper()}...")
    
    t0_custom = time.perf_counter()
    
    # 統一建構共用限制 (不能做空 + 30% 持股上限)
    G_no_short = -np.eye(n)
    h_no_short = np.zeros(n)
    
    G_max = np.eye(n)
    h_max = np.full(n, max_weight)
    
    G_base = np.vstack((G_no_short, G_max))
    h_base = np.concatenate((h_no_short, h_max))

    if args.solver == 'primal_dual':
        # Primal-Dual 擅長處理嚴格等式，預算限制使用 Ax = b
        A = np.ones((1, n))
        b = np.array([1.0])
        
        problem = QPProblem(Q=Q, c=c, G=G_base, h=h_base, A=A, b=b)
        solver = PrimalDualSolver(problem=problem, tol=1e-8)
        
        # 初始點: 均分權重 (確保嚴格滿足 Gx < h 且 Ax = b)
        # 1/6 約 0.166，落在 0 ~ 0.3 的安全區內
        x_init = np.ones(n) / n  
        
    else: # log_barrier
        # Log-Barrier 不支援等式，預算限制退化為不等式 sum(x) <= 1
        G_budget = np.ones((1, n))
        h_budget = np.array([1.0])
        
        G_full = np.vstack((G_base, G_budget))
        h_full = np.concatenate((h_base, h_budget))
        
        problem = QPProblem(Q=Q, c=c, G=G_full, h=h_full)
        solver = LogBarrierSolver(problem=problem)
        
        # 初始點: 確保嚴格滿足 sum(x) < 1 且 x < 0.3 且 x > 0
        x_init = np.ones(n) / (n + 1) 

    optimal_weights, info = solver.solve(x_init=x_init)
    t1_custom = time.perf_counter()
    time_custom = t1_custom - t0_custom

    # ==========================================
    # 5. 輸出 Performance Report
    # ==========================================
    print("\n" + "="*55)
    print(f"📊 OPTIMIZATION PERFORMANCE REPORT")
    print("="*55)
    print(f"{'Solver':<25} | {'Execution Time (sec)':<20} | {'Iters':<5}")
    print("-" * 55)
    print(f"{'CVXPY (OSQP Backend)':<25} | {time_cvxpy:<20.6f} | {'N/A':<5}")
    print(f"{'Custom ' + args.solver.upper():<25} | {time_custom:<20.6f} | {info.get('iterations', len(info.get('history_x', []))):<5}")
    print("="*55)
    
    # 比較兩者的最終權重差異
    print("\n--- Final Weights Comparison ---")
    print(f"{'Asset':<6} | {'CVXPY':<10} | {'Custom':<10} | {'Diff':<10}")
    print("-" * 42)
    for i, ticker in enumerate(tickers):
        w_cvx = cvxpy_weights[i]
        w_cus = optimal_weights[i]
        diff = abs(w_cvx - w_cus)
        print(f"{ticker:<6} | {w_cvx*100:>8.2f}% | {w_cus*100:>8.2f}% | {diff*100:>8.2e}%")

    # ==========================================
    # 6. 繪製圓餅圖 (使用 Custom Solver 的結果)
    # ==========================================
    allocation = {t: w for t, w in zip(tickers, optimal_weights) if w > 1e-4}
    
    plt.figure(figsize=(8, 6))
    plt.pie(allocation.values(), labels=allocation.keys(), autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title(f'Optimal Allocation (Solved via {args.solver.replace("_", " ").title()})', fontweight='bold')
    plt.axis('equal')  
    
    output_filename = "portfolio_allocation.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPie chart saved as {output_filename}")

if __name__ == "__main__":
    main()