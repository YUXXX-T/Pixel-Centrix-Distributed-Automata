import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局学术画图格式设置 (IEEE/ACM Standard)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],  # 学术常用无衬线字体
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# ==========================================
# 2. 实验数据输入 (从你的截图中提取)
# ==========================================
scenarios = ['10x10 Map\n(30 Robots)', '20x20 Map\n(120 Robots)']
x = np.arange(len(scenarios))  # X轴标签位置
width = 0.35  # 柱子宽度

# 图1数据：CPU计算总耗时 (秒)
time_pca = [2.93, 21.34]
time_pp = [9.27, 117.02]

# 图2数据：绝对吞吐量 (Deliveries / 500 Ticks)
thru_pca = [143, 270]
thru_pp = [263, 517]

# 图3数据：单位算力吞吐量 (Deliveries / CPU_second)
eff_pca = [48.81, 12.65]
eff_pp = [28.37, 4.42]

# 颜色设定：PCA用科技蓝（我们的主角），PP用次要灰/橘色（对比基准）
color_pca = '#1f77b4'  
color_pp = '#ff7f0e'   

# ==========================================
# 3. 创建 1x3 的并排子图
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# 辅助函数：在柱子顶部自动添加数值标签
def add_value_labels(ax, rects, is_float=False):
    for rect in rects:
        height = rect.get_height()
        label_text = f'{height:.2f}' if is_float else f'{int(height)}'
        ax.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移3个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

# --- 图 1：CPU 计算延迟 (越低越好) ---
rects1_pca = axes[0].bar(x - width/2, time_pca, width, label='PCA (Distributed)', color=color_pca, edgecolor='black', zorder=3)
rects1_pp = axes[0].bar(x + width/2, time_pp, width, label='PP (Centralized)', color=color_pp, edgecolor='black', zorder=3)
axes[0].set_ylabel('Total CPU Computation Time (s)')
axes[0].set_title('(a) Computational Delay vs. Scalability')
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenarios)
add_value_labels(axes[0], rects1_pca, is_float=True)
add_value_labels(axes[0], rects1_pp, is_float=True)
axes[0].legend()

# --- 图 2：物理绝对吞吐量 (越高越好，PP为理论上限) ---
rects2_pca = axes[1].bar(x - width/2, thru_pca, width, label='PCA', color=color_pca, edgecolor='black', zorder=3)
rects2_pp = axes[1].bar(x + width/2, thru_pp, width, label='PP (Upper Bound)', color=color_pp, edgecolor='black', zorder=3)
axes[1].set_ylabel('Total Deliveries (in 500 Ticks)')
axes[1].set_title('(b) Absolute Physical Throughput')
axes[1].set_xticks(x)
axes[1].set_xticklabels(scenarios)
add_value_labels(axes[1], rects2_pca)
add_value_labels(axes[1], rects2_pp)
axes[1].legend()

# --- 图 3：单位算力吞吐量 (王牌绝杀图，越高越好) ---
rects3_pca = axes[2].bar(x - width/2, eff_pca, width, label='PCA', color=color_pca, edgecolor='black', zorder=3)
rects3_pp = axes[2].bar(x + width/2, eff_pp, width, label='PP', color=color_pp, edgecolor='black', zorder=3)
axes[2].set_ylabel('Deliveries per CPU Second')
axes[2].set_title('(c) Computation-to-Performance Efficiency')
axes[2].set_xticks(x)
axes[2].set_xticklabels(scenarios)
add_value_labels(axes[2], rects3_pca, is_float=True)
add_value_labels(axes[2], rects3_pp, is_float=True)
axes[2].legend()

# ==========================================
# 4. 渲染并保存为高精度学术矢量图
# ==========================================
plt.tight_layout()

# 保存为 PDF 和高分辨率 PNG，PDF用于直接插入LaTeX
plt.savefig('./data/throughput_scalability_analysis.pdf', format='pdf', bbox_inches='tight')
plt.savefig('./data/throughput_scalability_analysis.png', format='png', dpi=300, bbox_inches='tight')

plt.show()