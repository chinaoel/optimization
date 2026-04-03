import numpy as np
import matplotlib.pyplot as plt

def plot_svm_decision_boundary(model, X, y, title="SVM Decision Boundary"):
    """
    視覺化 SVM 決策邊界與 Support Vectors。
    預期傳入的 model 必須有 predict() 方法，以及 support_vectors 屬性。
    """
    plt.figure(figsize=(8, 6))
    
    # 畫出原始資料點
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='o', label='+1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='x', label='-1')
    
    # 標示出 Support Vectors (畫黑圈)
    if hasattr(model, 'support_vectors') and model.support_vectors is not None:
        plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], 
                    s=100, linewidth=1.5, facecolors='none', edgecolors='k', 
                    label='Support Vectors')

    # 建立網格以繪製決策邊界與 Margin
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # 計算每個網格點的預測分數 (不是只有 1 或 -1，而是連續的距離分數)
    # 因為你的 predict 是回傳 np.sign，我們這裡需要手動算分數來畫等高線
    if model.alphas is not None:
        K_matrix = model.kernel(model.support_vectors, xy)
        Z = np.dot(model.sv_alphas * model.sv_y, K_matrix) + model.b
        Z = Z.reshape(XX.shape)
        
        # 畫出決策邊界 (Z=0) 與 Margin (Z=1, Z=-1)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

    plt.title(title)
    plt.legend()
    plt.show()