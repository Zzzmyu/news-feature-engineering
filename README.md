# News Feature Engineering & Dimensionality Reduction

> 大规模新闻文本特征工程与降维系统 | 课程项目

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 项目概述
处理CSCMNews新闻数据集（约33万文档，6个类别），实现从原始文本到低维特征的完整流水线：
- **文本预处理**：中文分词、停用词过滤、标点清理
- **特征提取**：TF-IDF向量化（构建5万维词典）
- **降维压缩**：TruncatedSVD维度压缩（200维，累计解释方差>80%）
- **可视化分析**：累计方差曲线、2D散点图、特征分布

## 🚀 快速开始
```bash
# 1. 克隆仓库
git clone https://github.com/yourname/news-feature-engineering.git
cd news-feature-engineering

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行完整流程
python scripts/run_pipeline.py --config config/config.yaml
