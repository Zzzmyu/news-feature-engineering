"""
可视化模块
生成分析图表
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
import os
from pathlib import Path

def plot_cumulative_variance(explained_variance_ratio: np.ndarray, 
                            save_path: Optional[str] = None,
                            figsize: tuple = (12, 8),
                            dpi: int = 300) -> plt.Figure:
    """
    绘制累计解释方差曲线
    
    Args:
        explained_variance_ratio: 解释方差比数组
        save_path: 保存路径
        figsize: 图像大小
        dpi: 分辨率
        
    Returns:
        matplotlib图形对象
    """
    cumulative = np.cumsum(explained_variance_ratio)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制累计曲线
    ax.plot(range(1, len(cumulative) + 1), cumulative, 
            'b-o', linewidth=2, markersize=4, label='累计解释方差')
    
    # 添加阈值线
    thresholds = [0.8, 0.9, 0.95]
    colors = ['orange', 'red', 'green']
    
    for thresh, color in zip(thresholds, colors):
        ax.axhline(y=thresh, color=color, linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'{thresh*100:.0f}% 阈值')
        
        # 找到对应的主成分数
        idx = np.argmax(cumulative >= thresh)
        if idx < len(cumulative):
            ax.axvline(x=idx + 1, color=color, linestyle=':', 
                      linewidth=1, alpha=0.5)
    
    ax.set_xlabel('主成分数量', fontsize=12)
    ax.set_ylabel('累计解释方差', fontsize=12)
    ax.set_title('累计解释方差曲线', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # 设置坐标轴范围
    ax.set_xlim(1, min(100, len(cumulative)))
    ax.set_ylim(0, 1.05)
    
    # 添加文本说明
    for i, thresh in enumerate(thresholds):
        n_components = np.argmax(cumulative >= thresh) + 1
        ax.text(0.02, 0.95 - i * 0.05, 
                f'达到{thresh*100:.0f}%方差需要{n_components}个主成分',
                transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"累计方差曲线已保存: {save_path}")
    
    return fig

def plot_scatter_2d(X_2d: np.ndarray, labels: List[str],
                   save_path: Optional[str] = None,
                   figsize: tuple = (14, 10),
                   dpi: int = 300,
                   alpha: float = 0.7,
                   max_categories: int = 10) -> plt.Figure:
    """
    绘制二维散点图
    
    Args:
        X_2d: 二维数据 (n_samples, 2)
        labels: 标签列表
        save_path: 保存路径
        figsize: 图像大小
        dpi: 分辨率
        alpha: 点透明度
        max_categories: 最多显示的分类数
        
    Returns:
        matplotlib图形对象
    """
    from collections import Counter
    
    # 统计类别
    label_counts = Counter(labels)
    
    # 选择最多的几个类别
    if len(label_counts) > max_categories:
        top_labels = [label for label, _ in label_counts.most_common(max_categories)]
        print(f"类别过多 ({len(label_counts)}个)，只显示前{max_categories}个")
    else:
        top_labels = list(label_counts.keys())
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 为每个类别分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_labels)))
    
    # 绘制散点
    for label, color in zip(top_labels, colors):
        # 获取该类别的索引
        indices = [i for i, l in enumerate(labels) if l == label]
        
        if indices:
            ax.scatter(X_2d[indices, 0], X_2d[indices, 1],
                      label=f'{label} ({len(indices)})',
                      c=[color], alpha=alpha, s=20,
                      edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('第一主成分 (PC1)', fontsize=12)
    ax.set_ylabel('第二主成分 (PC2)', fontsize=12)
    ax.set_title('PCA二维可视化', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    if len(top_labels) <= 15:
        ax.legend(loc='best', fontsize=10, ncol=2)
    else:
        ax.legend(loc='best', fontsize=8, ncol=3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"二维散点图已保存: {save_path}")
    
    return fig

def plot_variance_bar(explained_variance_ratio: np.ndarray,
                     n_components: int = 20,
                     save_path: Optional[str] = None,
                     figsize: tuple = (12, 6),
                     dpi: int = 300) -> plt.Figure:
    """
    绘制解释方差条形图
    
    Args:
        explained_variance_ratio: 解释方差比数组
        n_components: 显示的成分数
        save_path: 保存路径
        figsize: 图像大小
        dpi: 分辨率
        
    Returns:
        matplotlib图形对象
    """
    n_components = min(n_components, len(explained_variance_ratio))
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 创建条形图
    bars = ax.bar(range(1, n_components + 1), 
                  explained_variance_ratio[:n_components],
                  color='skyblue', alpha=0.7)
    
    # 添加累计曲线
    cumulative = np.cumsum(explained_variance_ratio[:n_components])
    ax2 = ax.twinx()
    ax2.plot(range(1, n_components + 1), cumulative,
            'r-o', linewidth=2, markersize=4, label='累计')
    
    ax.set_xlabel('主成分', fontsize=12)
    ax.set_ylabel('解释方差比', fontsize=12, color='blue')
    ax2.set_ylabel('累计解释方差', fontsize=12, color='red')
    
    ax.set_title(f'前{n_components}个主成分解释方差', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 设置颜色
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 添加图例
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"方差条形图已保存: {save_path}")
    
    return fig

def visualize_results(X_reduced: np.ndarray, labels: List[str],
                     reducer, output_dir: str, config: dict):
    """
    生成所有可视化结果
    
    Args:
        X_reduced: 降维后的数据
        labels: 标签列表
        reducer: 降维器对象
        output_dir: 输出目录
        config: 可视化配置
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    figsize = tuple(config.get('figure_size', [12, 8]))
    dpi = config.get('dpi', 300)
    save_format = config.get('save_format', 'png')
    
    # 1. 累计解释方差曲线
    if hasattr(reducer, 'explained_variance_ratio_') and reducer.explained_variance_ratio_ is not None:
        variance_path = os.path.join(output_dir, f'cumulative_variance.{save_format}')
        plot_cumulative_variance(
            reducer.explained_variance_ratio_,
            save_path=variance_path,
            figsize=figsize,
            dpi=dpi
        )
        plt.close()
    
    # 2. 二维散点图（使用前两个主成分）
    if X_reduced.shape[1] >= 2:
        scatter_path = os.path.join(output_dir, f'scatter_2d.{save_format}')
        plot_scatter_2d(
            X_reduced[:, :2],
            labels,
            save_path=scatter_path,
            figsize=figsize,
            dpi=dpi,
            alpha=0.7
        )
        plt.close()
    
    # 3. 解释方差条形图
    if hasattr(reducer, 'explained_variance_ratio_') and reducer.explained_variance_ratio_ is not None:
        bar_path = os.path.join(output_dir, f'variance_bar.{save_format}')
        plot_variance_bar(
            reducer.explained_variance_ratio
