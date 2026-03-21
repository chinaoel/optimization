import os
import sys
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.qp_problem import QPProblem
from core.primal_dual import PrimalDualSolver

def get_market_data(tickers: list, start_date: str, end_date: str):
    print(f"Fetching data for: {', '.join(tickers)}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data[tickers]
    
    returns = data.pct_change().dropna()
    trading_days = 252
    
    annual_returns = returns.mean() * trading_days
    annual_cov = returns.cov() * trading_days
    
    return annual_returns.values, annual_cov.values

def main():
    print("=== Generating Efficient Frontier ===")
    
    # 1. 取得數據
    tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', # 科技五巨頭 (高報酬/高波動)
    'BRK-B', 'JPM',                          # 金融巨頭 (價值型/穩定)
    'JNJ', 'PG',                             # 防禦型 (抗跌)
    'TLT', 'GLD'                             # 絕對避險 (美債與黃金)
    ]
    expected_returns, cov_matrix = get_market_data(tickers, "2020-01-01", "2024-01-01")
    n = len(tickers)
    Q = cov_matrix

    # 2. 定義共用約束條件 (不能做空 + 30% 持股上限 + 權重和為1)
    G_no_short = -np.eye(n)
    h_no_short = np.zeros(n)
    G_max = np.eye(n)
    h_max = np.full(n, 0.30)
    G = np.vstack((G_no_short, G_max))
    h = np.concatenate((h_no_short, h_max))
    A = np.ones((1, n))
    b = np.array([1.0])

    # 3. 掃描不同的風險厭惡係數 (lambda)
    # 使用 logspace 產生從 10^-3 到 10^2 的 50 個點，讓曲線分佈更均勻
    lambdas = np.logspace(-3, 2, 50)
    
    portfolio_vols = []
    portfolio_rets = []
    
    x_init = np.ones(n) / n
    print(f"Solving {len(lambdas)} QP problems to trace the frontier...")
    
    for lmbda in lambdas:
        # 更新目標函數中的 c 向量: c = -lambda * mu
        c = -lmbda * expected_returns
        
        problem = QPProblem(Q=Q, c=c, G=G, h=h, A=A, b=b)
        solver = PrimalDualSolver(problem=problem, tol=1e-8)
        
        # 我們將前一次的解當作下一次的初始點 (Warm Start)，收斂會更快！
        optimal_weights, info = solver.solve(x_init=x_init, max_iters=50)
        x_init = optimal_weights 
        
        # 計算該最佳組合的年化波動率與報酬率
        port_variance = optimal_weights.T @ Q @ optimal_weights
        port_volatility = np.sqrt(port_variance)
        port_return = expected_returns.T @ optimal_weights
        
        portfolio_vols.append(port_volatility)
        portfolio_rets.append(port_return)

    # 4. 計算單一資產的波動率與報酬率 (作為圖表對比用)
    asset_vols = np.sqrt(np.diag(cov_matrix))
    asset_rets = expected_returns

    # 5. 繪製精美的效率前緣圖表
    plt.figure(figsize=(10, 6))
    
    # 畫出前緣曲線
    plt.plot(portfolio_vols, portfolio_rets, 'b-', linewidth=3, label='Constrained Efficient Frontier (Max 30% per asset)')
    
    # 畫出單一資產的散點
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i in range(n):
        plt.scatter(asset_vols[i], asset_rets[i], color=colors[i], s=100, zorder=5)
        plt.annotate(tickers[i], (asset_vols[i], asset_rets[i]), 
                     xytext=(5, 5), textcoords='offset points', 
                     fontsize=11, fontweight='bold')
        
    plt.title('Markowitz Efficient Frontier\nComputed by Custom Primal-Dual IPM', fontsize=14, fontweight='bold')
    plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
    plt.ylabel('Annualized Expected Return', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=11)
    
    # 將 Y 軸與 X 軸轉成百分比顯示
    import matplotlib.ticker as mtick
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig('efficient_frontier.png', dpi=300)
    print("Frontier generated successfully! Saved as 'efficient_frontier.png'")

if __name__ == "__main__":
    main()