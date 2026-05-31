"""
向量化（Embedding）模块
-----------------------
将文本转换为固定维度的浮点数向量，供 FAISS 进行相似度检索。

默认使用 sentence-transformers 本地模型（免费、隐私安全、离线可用）。
已预留 APIEmbedding 接口，后续可接入千问/DeepSeek 等云端 Embedding API。

模型选择建议：
  - 中英混合：paraphrase-multilingual-MiniLM-L12-v2  （384维，轻量快速）
  - 纯中文　：BAAI/bge-small-zh-v1.5                 （512维，中文效果更好）

扩展接口：
  实现新的 Embedding 类，只需继承 BaseEmbedding 并实现 embed_texts() 方法。
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from config.settings import get_settings


# ========================================================================
# 抽象基类
# ========================================================================

class BaseEmbedding(ABC):
    """
    Embedding 抽象基类。
    所有向量化实现必须提供 embed_texts() 方法。
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量矩阵。

        Args:
            texts: 待向量化的文本列表

        Returns:
            np.ndarray: shape = (len(texts), dim) 的向量矩阵
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """返回向量维度。"""
        ...


# ========================================================================
# 本地 Sentence-Transformer Embedding
# ========================================================================

class LocalEmbedding(BaseEmbedding):
    """
    基于 sentence-transformers 的本地向量化。

    首次运行时会自动下载模型（约 80~500 MB），
    之后缓存在本地，离线可用。
    """

    def __init__(
        self,
        model_name: Union[str, None] = None,
        device: Union[str, None] = None,
    ):
        """
        Args:
            model_name: HuggingFace 模型名，默认从配置读取
            device: 运行设备（cpu / cuda / mps），默认从配置读取
        """
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self._model_name = model_name or settings.LOCAL_EMBEDDING_MODEL
        self._device = device or settings.EMBEDDING_DEVICE

        print(f"  加载 Embedding 模型: {self._model_name}")
        print(f"  运行设备: {self._device}")

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

    @property
    def dim(self) -> int:
        """向量维度（由模型决定）。"""
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        批量向量化文本。

        Args:
            texts: 文本列表

        Returns:
            shape=(len(texts), dim) 的 float32 向量矩阵
        """
        if not texts:
            return np.array([], dtype=np.float32)

        # sentence-transformers 内部自动分批处理
        # 仅在批量较大时显示进度条（避免与 stdin pipe 冲突导致 tokenizer 类型错误）
        show_bar = len(texts) > 10
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化 → 可直接用内积做余弦相似度
            show_progress_bar=show_bar,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        对单条查询文本进行向量化（便捷方法）。

        Args:
            text: 查询文本

        Returns:
            shape=(dim,) 的向量
        """
        vec = self.embed_texts([text])
        return vec[0]


# ========================================================================
# API Embedding（预留扩展接口，后续接入千问等）
# ========================================================================

class APIEmbedding(BaseEmbedding):
    """
    基于 API 的 Embedding 服务（预留）。

    后续接入千问 AI 或其他云端 Embedding 时，在此实现 embed_texts()。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        # TODO: 接入具体 API 客户端
        raise NotImplementedError("APIEmbedding 尚未实现，请使用 LocalEmbedding")

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


# ========================================================================
# 工厂函数
# ========================================================================

# 全局 Embedding 实例缓存（单例）
_embedding_instance: Union[BaseEmbedding, None] = None


def get_embedding(
    embedding_type: str = "local",
    **kwargs,
) -> BaseEmbedding:
    """
    获取 Embedding 实例（单例缓存，避免重复加载模型）。

    Args:
        embedding_type: "local" 或 "api"
        **kwargs: 传递给具体 Embedding 类的参数

    Returns:
        BaseEmbedding 实例
    """
    global _embedding_instance

    if _embedding_instance is not None:
        return _embedding_instance

    if embedding_type == "local":
        _embedding_instance = LocalEmbedding(**kwargs)
    elif embedding_type == "api":
        _embedding_instance = APIEmbedding(**kwargs)
    else:
        raise ValueError(f"不支持的 embedding_type: {embedding_type}")

    return _embedding_instance
