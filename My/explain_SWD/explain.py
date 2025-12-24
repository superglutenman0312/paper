import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def draw_swd_process_diagram():
    # 設定畫布大小 (寬一點，適合放進橫向的架構圖中)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 定義顏色
    c_source = "#F5D60C" # RoyalBlue (Source)
    c_target = "#FA7705" # IndianRed (Target)
    c_match = '#32CD32'  # LimeGreen (WD Link)
    c_proj = 'gray'
    
    # ==========================================
    # 步驟 1: High-dimensional Feature Space
    # ==========================================
    ax = axes[0]
    ax.set_title("1: Feature Extractor Output\n(64-dim Point Cloud)", fontsize=16, pad=20)
    
    # 模擬 2D 上的高維分佈 (稍微分開的兩團)
    np.random.seed(42)
    s_data = np.random.randn(20, 2) * 0.5 + [1, 1]
    t_data = np.random.randn(20, 2) * 0.5 + [2.5, 2.5]
    
    ax.scatter(s_data[:,0], s_data[:,1], c=c_source, s=100, alpha=0.8, label='Source')
    ax.scatter(t_data[:,0], t_data[:,1], c=c_target, s=100, alpha=0.8, label='Target')
    
    # 畫個虛擬的外框表示 Feature Space
    rect = patches.Rectangle((-1, -1), 6, 6, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    ax.text(0, 4, "High-dim Space", fontsize=12, style='italic')
    ax.set_xlim(-1.5, 5)
    ax.set_ylim(-1.5, 5)
    ax.axis('off') # 隱藏座標軸

    # ==========================================
    # 步驟 2: Random Projection (Slicing)
    # ==========================================
    ax = axes[1]
    ax.set_title("2: Random Projections\n(Generate 50 Slices)", fontsize=16, pad=20)
    
    # 畫單位圓
    circle = patches.Circle((0, 0), radius=1, edgecolor='black', facecolor='#f0f0f0', alpha=0.5, linestyle='-')
    ax.add_patch(circle)
    
    # 畫幾條隨機的投影線 (Slices)
    angles = [30, 75, 120, 160]
    for deg in angles:
        rad = np.radians(deg)
        x = np.cos(rad) * 1.2
        y = np.sin(rad) * 1.2
        ax.plot([-x, x], [-y, y], color=c_proj, linestyle='--', lw=2)
        
    ax.text(0, -1.5, "Project Features\nonto lines", ha='center', fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.8, 1.5)
    ax.axis('off')

    # ==========================================
    # 步驟 3: Sorting & 1D WD Calculation
    # ==========================================
    ax = axes[2]
    ax.set_title("3: Sort & Measure 1D Distance\n(On each slice)", fontsize=16, pad=20)
    
    # 模擬 1D 數據 (排序後)
    y_gap = 0.5
    n_pts = 6
    s_1d = np.sort(np.random.uniform(0, 5, n_pts))
    t_1d = np.sort(np.random.uniform(1, 6, n_pts))
    
    # 畫兩條軌道 (Source 上, Target 下)
    ax.plot([0, 6], [y_gap, y_gap], color=c_source, lw=1, alpha=0.3)
    ax.plot([0, 6], [-y_gap, -y_gap], color=c_target, lw=1, alpha=0.3)
    
    # 畫點
    ax.scatter(s_1d, [y_gap]*n_pts, c=c_source, s=80, zorder=3)
    ax.scatter(t_1d, [-y_gap]*n_pts, c=c_target, s=80, zorder=3)
    
    # 畫 WD 連線 (綠色)
    for s, t in zip(s_1d, t_1d):
        ax.plot([s, t], [y_gap, -y_gap], c=c_match, lw=3, alpha=0.8)

    ax.text(3, 0.8, "Sorted Source", ha='center', color=c_source, fontsize=12, fontweight='bold')
    ax.text(3, -0.9, "Sorted Target", ha='center', color=c_target, fontsize=12, fontweight='bold')
    # ax.text(3, 0, "Wasserstein Dist.", ha='center', color=c_match, fontsize=10, backgroundcolor='white')
    
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    # ==========================================
    # 步驟 4: Aggregation (Output)
    # ==========================================
    ax = axes[3]
    ax.set_title("4: Domain Loss Aggregation\n(Average all slices)", fontsize=16, pad=20)
    
    # 畫數學公式示意
    ax.text(0.5, 0.6, r"$L_{SWD} = \frac{1}{N}\sum_{i=1}^{N} W_2(\theta_i)$", 
            ha='center', va='center', fontsize=24, color='black')
            
    # 畫最終輸出箭頭
    ax.arrow(0.5, 0.3, 0, -0.2, head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    # 畫 Loss 圖示
    circle_loss = patches.Circle((0.5, -0.2), radius=0.15, edgecolor='black', facecolor=c_match)
    ax.add_patch(circle_loss)
    ax.text(0.5, -0.2, "Loss", ha='center', va='center', color='white', fontweight='bold')

    ax.axis('off')
    
    # ==========================================
    # 添加步驟間的箭頭 (Flow Arrows)
    # ==========================================
    # 利用 figure 的座標系 (0~1) 來畫跨子圖的箭頭
    # 這裡微調箭頭位置，讓它看起來像一個連貫的流程
    arrow_y = 0.5
    plt.figtext(0.28, arrow_y, "➜", fontsize=40, ha='center', color='gray') # Step 1 to 2
    plt.figtext(0.51, arrow_y, "➜", fontsize=40, ha='center', color='gray') # Step 2 to 3
    plt.figtext(0.73, arrow_y, "➜", fontsize=40, ha='center', color='gray') # Step 3 to 4

    plt.tight_layout()
    plt.show()

# 執行繪圖
draw_swd_process_diagram()