"""
文本预处理模块
中文分词、停用词过滤、文本清洗
"""
import re
import jieba
from typing import List, Set
import os

class TextProcessor:
    """文本处理器"""
    
    def __init__(self, stopwords_path: str = None):
        """
        初始化文本处理器
        
        Args:
            stopwords_path: 停用词文件路径
        """
        self.stopwords: Set[str] = set()
        if stopwords_path and os.path.exists(stopwords_path):
            self.stopwords = self._load_stopwords(stopwords_path)
            print(f"加载停用词: {len(self.stopwords)} 个")
        else:
            print("未提供停用词文件，将不使用停用词过滤")
    
    def _load_stopwords(self, path: str) -> Set[str]:
        """加载停用词表"""
        stopwords = set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
        except Exception as e:
            print(f"加载停用词失败: {e}")
        return stopwords
    
    def preprocess(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        文本预处理：分词、清洗、过滤
        
        Args:
            text: 原始文本
            remove_stopwords: 是否移除停用词
            
        Returns:
            处理后的词语列表
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 1. 基础清洗
        text = text.lower()  # 英文转小写
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)  # 去除非中英文字符和数字
        text = re.sub(r'\s+', ' ', text)  # 合并空白字符
        
        # 2. 分词
        words = jieba.lcut(text)
        
        # 3. 过滤
        filtered_words = []
        for word in words:
            word = word.strip()
            if not word or len(word) < 2:  # 过滤空词和单字
                continue
            if remove_stopwords and word in self.stopwords:  # 过滤停用词
                continue
            filtered_words.append(word)
        
        return filtered_words
    
    def preprocess_batch(self, texts: List[str], remove_stopwords: bool = True) -> List[List[str]]:
        """
        批量文本预处理
        
        Args:
            texts: 文本列表
            remove_stopwords: 是否移除停用词
            
        Returns:
            处理后的词语列表的列表
        """
        return [self.preprocess(text, remove_stopwords) for text in texts]
    
    def preprocess_to_text(self, text: str, remove_stopwords: bool = True) -> str:
        """
        预处理并返回字符串
        
        Args:
            text: 原始文本
            remove_stopwords: 是否移除停用词
            
        Returns:
            预处理后的文本字符串
        """
        words = self.preprocess(text, remove_stopwords)
        return ' '.join(words)
