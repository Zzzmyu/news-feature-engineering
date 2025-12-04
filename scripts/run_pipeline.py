#!/usr/bin/env python
"""
主运行脚本 - 特征工程流水线
"""
import argparse
import yaml
import time
import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessor import TextProcessor
from vectorizer import FeatureVectorizer
from reducer import DimensionalityReducer
from visualization import visualize_results  # 会在后面创建

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_sample_data(n_samples: int = 1000) -> tuple:
    """
    创建示例数据
    实际项目中应该从文件加载真实数据
    """
    print(f"生成 {n_samples} 个示例文本...")
    
    # 示例文本（实际应该从文件加载）
    sample_texts = [
        "自然语言处理是人工智能的重要分支",
        "深度学习在计算机视觉领域取得突破",
        "机器学习算法需要大量数据进行训练",
        "特征工程是机器学习的关键步骤",
        "文本分类需要好的特征表示方法",
        "中文分词是中文NLP的基础任务",
        "情感分析可以判断文本的情感倾向",
        "命名实体识别用于识别文本中的实体",
        "文本摘要可以自动生成文章摘要",
        "机器翻译实现不同语言之间的转换"
    ]
    
    # 扩展样本
    texts = []
    labels = []
    for i in range(n_samples):
        text = sample_texts[i % len(sample_texts)] + f" 样本编号 {i}"
        texts.append(text)
        labels.append(f"类别_{i % 6}")  # 6个类别
    
    return texts, labels

def run_pipeline(config: dict):
    """运行特征工程流水线"""
    print("=" * 60)
    print("🚀 开始特征工程流水线")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # 1. 初始化组件
    print("\n1. 初始化组件...")
    processor = TextProcessor(stopwords_path=config['data']['stopwords_path'])
    vectorizer = FeatureVectorizer(
        max_features=config['vectorization']['max_features'],
        max_df=config['vectorization']['max_df'],
        min_df=config['vectorization']['min_df'],
        norm=config['vectorization']['norm'],
        use_idf=config['vectorization']['use_idf'],
        sublinear_tf=config['vectorization']['sublinear_tf']
    )
    reducer = DimensionalityReducer(
        n_components=config['dimensionality_reduction']['n_components'],
        method=config['dimensionality_reduction']['method'],
        random_state=config['dimensionality_reduction']['random_state'],
        batch_size=config['dimensionality_reduction'].get('batch_size', 1000)
    )
    
    # 2. 加载数据（这里使用示例数据）
    print("\n2. 加载数据...")
    texts, labels = create_sample_data(config['data']['sample_size'])
    print(f"   加载 {len(texts)} 个文档，{len(set(labels))} 个类别")
    
    # 3. 文本预处理
    print("\n3. 文本预处理...")
    preprocess_start = time.time()
    
    processed_texts = []
    for i, text in enumerate(texts):
        processed = processor.preprocess_to_text(
            text, 
            remove_stopwords=config['preprocessing']['remove_stopwords']
        )
        processed_texts.append(processed)
        
        # 显示进度
        if (i + 1) % 1000 == 0:
            print(f"   已处理 {i + 1}/{len(texts)} 个文档")
    
    preprocess_time = time.time() - preprocess_start
    print(f"   预处理完成，耗时: {preprocess_time:.2f}秒")
    
    # 4. 特征提取
    print("\n4. 特征提取 (TF-IDF)...")
    vectorize_start = time.time()
    
    X, vocabulary = vectorizer.fit_transform(processed_texts)
    
    vectorize_time = time.time() - vectorize_start
    print(f"   特征提取完成，耗时: {vectorize_time:.2f}秒")
    print(f"   特征维度: {X.shape[1]}")
    
    # 5. 降维
    print("\n5. 维度压缩...")
    reduce_start = time.time()
    
    X_reduced = reducer.fit_transform(X)
    
    reduce_time = time.time() - reduce_start
    print(f"   降维完成，耗时: {reduce_time:.2f}秒")
    print(f"   目标维度: {X_reduced.shape[1]}")
    
    # 6. 分析结果
    print("\n6. 结果分析...")
    
    # 累计解释方差
    cumulative_var = reducer.get_cumulative_variance()[-1]
    print(f"   累计解释方差: {cumulative_var:.4f}")
    
    # 达到90%方差所需维度
    if reducer.explained_variance_ratio_ is not None:
        n_for_90 = reducer.get_variance_threshold(0.9)
        print(f"   达到90%方差所需维度: {n_for_90}")
    
    # 7. 可视化
    if config['output']['visualize']:
        print("\n7. 生成可视化...")
        try:
            # 创建输出目录
            output_dir = config['output']['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成可视化
            visualize_results(
                X_reduced=X_reduced,
                labels=labels,
                reducer=reducer,
                output_dir=output_dir,
                config=config['output']['visualization']
            )
            print(f"   可视化已保存到: {output_dir}")
        except Exception as e:
            print(f"   可视化生成失败: {e}")
    
    # 8. 保存结果
    if config['output']['save_features']:
        print("\n8. 保存结果...")
        output_dir = config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存降维后的特征
        if 'npy' in config['output']['formats']:
            np.save(os.path.join(output_dir, 'features_reduced.npy'), X_reduced)
            print(f"   保存为NumPy格式: features_reduced.npy")
        
        # 保存标签
        np.save(os.path.join(output_dir, 'labels.npy'), np.array(labels))
        
        # 保存词汇表
        if vocabulary is not None:
            np.save(os.path.join(output_dir, 'vocabulary.npy'), vocabulary)
        
        # 保存模型
        vectorizer.save(os.path.join(output_dir, 'vectorizer.pkl'))
        reducer.save(os.path.join(output_dir, 'reducer.pkl'))
    
    # 9. 统计信息
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("✅ 流水线完成!")
    print("=" * 60)
    print(f"总耗时: {total_time:.2f}秒")
    print(f"原始维度: {X.shape[1]} → 降维后: {X_reduced.shape[1]}")
    print(f"压缩比例: {(1 - X_reduced.shape[1] / X.shape[1]) * 100:.1f}%")
    print(f"累计解释方差: {cumulative_var:.4f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文本特征工程流水线')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径 (默认: config/config.yaml)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='样本数量 (覆盖配置文件中的设置)')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print("请确保 config/config.yaml 文件存在")
        sys.exit(1)
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖样本数量（如果提供了命令行参数）
    if args.sample_size:
        config['data']['sample_size'] = args.sample_size
    
    # 运行流水线
    try:
        run_pipeline(config)
    except Exception as e:
        print(f"流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 需要numpy，但只在运行时导入
    import numpy as np
    main()
