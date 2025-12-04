"""
特征向量化模块
TF-IDF特征提取与稀疏矩阵处理
"""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Optional
import pickle
import os

class FeatureVectorizer:
    """特征向量化器"""
    
    def __init__(self, max_features: int = 50000, **kwargs):
        """
        初始化向量化器
        
        Args:
            max_features: 最大特征数
            **kwargs: 传递给TfidfVectorizer的参数
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=kwargs.get('max_df', 0.8),
            min_df=kwargs.get('min_df', 5),
            norm=kwargs.get('norm', 'l2'),
            use_idf=kwargs.get('use_idf', True),
            sublinear_tf=kwargs.get('sublinear_tf', True)
        )
        self.vocabulary_: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'FeatureVectorizer':
        """
        训练向量化器
        
        Args:
            texts: 文本列表
            
        Returns:
            self
        """
        print(f"训练TF-IDF向量化器，最大特征数: {self.max_features}")
        self.vectorizer.fit(texts)
        self.vocabulary_ = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        print(f"训练完成，词典大小: {len(self.vocabulary_)}")
        return self
    
    def transform(self, texts: List[str]) -> sparse.csr_matrix:
        """
        转换文本为TF-IDF矩阵
        
        Args:
            texts: 文本列表
            
        Returns:
            稀疏TF-IDF矩阵
        """
        if not self.is_fitted:
            raise ValueError("向量化器未训练，请先调用fit方法")
        
        print(f"转换 {len(texts)} 个文档...")
        X = self.vectorizer.transform(texts)
        
        # 计算稀疏率
        sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
        print(f"转换完成，矩阵形状: {X.shape}, 稀疏率: {sparsity:.4f}")
        
        return X
    
    def fit_transform(self, texts: List[str]) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        训练并转换
        
        Args:
            texts: 文本列表
            
        Returns:
            (TF-IDF矩阵, 词汇表)
        """
        X = self.vectorizer.fit_transform(texts)
        self.vocabulary_ = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        return X, self.vocabulary_
    
    def save(self, path: str) -> None:
        """保存向量化器"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'vocabulary': self.vocabulary_,
                'max_features': self.max_features,
                'is_fitted': self.is_fitted
            }, f)
        print(f"向量化器已保存到: {path}")
    
    def load(self, path: str) -> 'FeatureVectorizer':
        """加载向量化器"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.vocabulary_ = data['vocabulary']
        self.max_features = data['max_features']
        self.is_fitted = data['is_fitted']
        
        print(f"向量化器已加载，词典大小: {len(self.vocabulary_)}")
        return self
    
    def get_feature_names(self) -> np.ndarray:
        """获取特征名称"""
        if self.vocabulary_ is None:
            raise ValueError("向量化器未训练")
        return self.vocabulary_
    
    def analyze_text(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        分析文本的特征权重
        
        Args:
            text: 文本
            top_n: 显示前N个特征
            
        Returns:
            (特征词, 权重) 列表
        """
        if not self.is_fitted:
            raise ValueError("向量化器未训练")
        
        # 转换为TF-IDF向量
        vector = self.vectorizer.transform([text])
        
        # 获取非零特征
        indices = vector.indices
        values = vector.data
        
        # 按权重排序
        sorted_indices = np.argsort(values)[::-1][:top_n]
        results = []
        
        for idx in sorted_indices:
            feature_idx = indices[idx]
            weight = values[idx]
            feature_name = self.vocabulary_[feature_idx]
            results.append((feature_name, weight))
        
        return results
