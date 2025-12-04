"""
降维模块
PCA与TruncatedSVD降维实现
"""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA
from typing import Union, Optional
import pickle
import os

class DimensionalityReducer:
    """降维器"""
    
    def __init__(self, n_components: int = 200, method: str = 'truncated_svd', **kwargs):
        """
        初始化降维器
        
        Args:
            n_components: 目标维度
            method: 降维方法 ('pca', 'truncated_svd', 'incremental_pca')
            **kwargs: 传递给降维算法的参数
        """
        self.n_components = n_components
        self.method = method
        self.model = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.is_fitted = False
        
        # 初始化模型
        if method == 'truncated_svd':
            self.model = TruncatedSVD(
                n_components=n_components,
                random_state=kwargs.get('random_state', 42),
                n_iter=kwargs.get('n_iter', 7)
            )
        elif method == 'pca':
            self.model = PCA(
                n_components=n_components,
                random_state=kwargs.get('random_state', 42)
            )
        elif method == 'incremental_pca':
            self.model = IncrementalPCA(
                n_components=n_components,
                batch_size=kwargs.get('batch_size', 1000)
            )
        else:
            raise ValueError(f"不支持的降维方法: {method}")
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix]) -> 'DimensionalityReducer':
        """
        训练降维模型
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            self
        """
        print(f"训练降维模型，方法: {self.method}, 目标维度: {self.n_components}")
        print(f"输入数据形状: {X.shape}")
        
        self.model.fit(X)
        
        # 获取解释方差
        if hasattr(self.model, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.model.explained_variance_ratio_
            cumulative_var = self.get_cumulative_variance()[-1]
            print(f"训练完成，累计解释方差: {cumulative_var:.4f}")
        else:
            print("训练完成")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """
        降维转换
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            降维后的数据
        """
        if not self.is_fitted:
            raise ValueError("降维模型未训练，请先调用fit方法")
        
        print(f"降维转换，输入形状: {X.shape} -> 输出形状: ({X.shape[0]}, {self.n_components})")
        return self.model.transform(X)
    
    def fit_transform(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """
        训练并转换
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            降维后的数据
        """
        return self.fit(X).transform(X)
    
    def get_cumulative_variance(self) -> np.ndarray:
        """
        获取累计解释方差
        
        Returns:
            累计解释方差数组
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("模型未训练，无法获取解释方差")
        return np.cumsum(self.explained_variance_ratio_)
    
    def get_variance_threshold(self, threshold: float = 0.9) -> int:
        """
        获取达到指定解释方差所需的主成分数
        
        Args:
            threshold: 解释方差阈值 (0-1)
            
        Returns:
            所需主成分数
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("模型未训练")
        
        cumulative = self.get_cumulative_variance()
        for i, var in enumerate(cumulative):
            if var >= threshold:
                return i + 1
        
        return len(cumulative)
    
    def save(self, path: str) -> None:
        """保存降维模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'n_components': self.n_components,
                'method': self.method,
                'explained_variance_ratio': self.explained_variance_ratio_,
                'is_fitted': self.is_fitted
            }, f)
        print(f"降维模型已保存到: {path}")
    
    def load(self, path: str) -> 'DimensionalityReducer':
        """加载降维模型"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.n_components = data['n_components']
        self.method = data['method']
        self.explained_variance_ratio_ = data['explained_variance_ratio']
        self.is_fitted = data['is_fitted']
        
        print(f"降维模型已加载，方法: {self.method}, 维度: {self.n_components}")
        return self
    
    def analyze_components(self, feature_names: Optional[np.ndarray] = None, 
                          n_components: int = 5, n_features: int = 10) -> dict:
        """
        分析主成分
        
        Args:
            feature_names: 特征名称数组
            n_components: 分析前N个主成分
            n_features: 每个主成分显示前N个特征
            
        Returns:
            主成分分析结果
        """
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        if not hasattr(self.model, 'components_'):
            raise ValueError(f"{self.method} 不支持成分分析")
        
        components = self.model.components_
        n_components = min(n_components, components.shape[0])
        
        results = {}
        
        for i in range(n_components):
            component = components[i]
            
            # 获取最重要的特征
            if feature_names is not None:
                top_indices = np.argsort(np.abs(component))[::-1][:n_features]
                top_features = []
                
                for idx in top_indices:
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        weight = component[idx]
                        top_features.append((feature_name, weight))
                
                results[f'PC{i+1}'] = {
                    'explained_variance': self.explained_variance_ratio_[i] if self.explained_variance_ratio_ is not None else None,
                    'top_features': top_features
                }
        
        return results
