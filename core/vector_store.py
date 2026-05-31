"""
向量库存储模块
-------------
基于 FAISS 实现本地向量索引的构建、持久化和检索。

技术细节：
  - 使用 IndexIDMap 包装 IndexFlatIP（内积），等价于余弦相似度（配合 L2 归一化向量）
  - 每个向量分配一个唯一 ID（自增整数），通过 ID 映射到原始 chunk 文本
  - 索引可持久化到磁盘，下次启动时直接加载，无需重新构建

扩展接口：
  如需切换向量库（如 Milvus、Chroma），只需实现相同接口的 Store 类。
"""

import os
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
import faiss

from core.document_loader import Document
from core.embeddings import BaseEmbedding


class FAISSVectorStore:
    """
    FAISS 向量库管理器。

    职责：
      1. 将 Document 列表向量化后写入 FAISS 索引
      2. 相似度检索（返回最相关的 k 个 chunk）
      3. 索引持久化（save / load）

    Attributes:
        index          : FAISS 索引对象
        id_to_document : {向量ID → Document} 映射表
        next_id        : 下一个可用 ID
        dim            : 向量维度
    """

    def __init__(self, embedding: BaseEmbedding):
        """
        Args:
            embedding: Embedding 实例（用于将文本转为向量）
        """
        self.embedding = embedding
        self.dim = embedding.dim
        self.index = None
        self.id_to_document: Dict[int, Document] = {}
        self.next_id = 0

    # ========================================================================
    # 索引构建
    # ========================================================================

    def build_from_documents(self, documents: List[Document]):
        """
        从 Document 列表构建 FAISS 索引。

        流程：
          1. 提取所有 Document 的 content 文本
          2. 调用 Embedding 模型批量向量化
          3. 创建 FAISS 索引并添加向量
          4. 建立 ID → Document 映射

        Args:
            documents: 切片后的 Document 列表
        """
        if not documents:
            raise ValueError("document 列表为空，无法构建索引")

        print(f"\n  正在为 {len(documents)} 个文档片段生成向量...")

        # 提取文本
        texts = [doc.content for doc in documents]

        # 批量向量化
        vectors = self.embedding.embed_texts(texts)

        # 创建 FAISS 索引
        self._create_index(vectors)

        # 建立 ID 映射
        self.id_to_document = {}
        for i, doc in enumerate(documents):
            self.id_to_document[i] = doc
        self.next_id = len(documents)

        print(f"  索引构建完成，共 {self.index.ntotal} 条向量，维度 {self.dim}")

    def _create_index(self, vectors: np.ndarray):
        """
        创建 FAISS 索引。

        使用 IndexIDMap(IndexFlatIP) 架构：
          - IndexFlatIP：暴力内积搜索（精确但 O(N)）
          - IndexIDMap：支持自定义 ID 而非默认的 0,1,2,...

        配合 L2 归一化向量，内积等价于余弦相似度。
        """
        # 确保向量是 float32 且连续存储（FAISS 要求）
        vectors = np.ascontiguousarray(vectors.astype(np.float32))

        # 底层索引：内积搜索
        base_index = faiss.IndexFlatIP(self.dim)

        # 包装为 ID 映射索引
        self.index = faiss.IndexIDMap(base_index)

        # 生成 ID 序列
        ids = np.arange(vectors.shape[0], dtype=np.int64)

        # 添加向量
        self.index.add_with_ids(vectors, ids)

    # ========================================================================
    # 检索
    # ========================================================================

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        向量相似度检索。

        Args:
            query_vector: 查询向量，shape=(dim,)
            top_k: 返回最相似的 k 条结果

        Returns:
            [(Document, score), ...] 按相似度降序排列
            score 为余弦相似度（范围 -1 ~ 1，越大越相关）
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # FAISS search 要求 2D 输入: (1, dim)
        query = np.ascontiguousarray(
            query_vector.astype(np.float32).reshape(1, -1)
        )

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS 占位符，无实际结果
            doc = self.id_to_document.get(int(idx))
            if doc is not None:
                results.append((doc, float(score)))

        return results

    # ========================================================================
    # 持久化（保存 / 加载）
    # ========================================================================

    def save(self, directory: str):
        """
        将索引和元数据保存到磁盘。

        生成两个文件：
          - index.faiss  : FAISS 索引二进制文件
          - metadata.pkl : ID→Document 映射表

        Args:
            directory: 保存目录路径
        """
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.pkl")

        # 保存 FAISS 索引
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            print(f"  索引已保存: {index_path} ({self.index.ntotal} 条)")

        # 保存元数据（ID→Document 映射）
        with open(meta_path, "wb") as f:
            pickle.dump({
                "id_to_document": self.id_to_document,
                "next_id": self.next_id,
            }, f)
        print(f"  元数据已保存: {meta_path}")

    def load(self, directory: str) -> bool:
        """
        从磁盘加载索引和元数据。

        Args:
            directory: 索引文件所在目录

        Returns:
            bool: 加载成功返回 True，文件不存在返回 False
        """
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        # 加载 FAISS 索引
        self.index = faiss.read_index(index_path)
        self.dim = self.index.d  # 从索引中恢复维度

        # 加载元数据
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            self.id_to_document = meta["id_to_document"]
            self.next_id = meta["next_id"]

        print(f"  索引已加载: {index_path} ({self.index.ntotal} 条向量)")
        return True

    # ========================================================================
    # 信息查询
    # ========================================================================

    @property
    def count(self) -> int:
        """索引中的向量总数。"""
        return self.index.ntotal if self.index is not None else 0

    @property
    def is_empty(self) -> bool:
        """索引是否为空。"""
        return self.count == 0

    def get_stats(self) -> dict:
        """返回索引统计信息。"""
        return {
            "total_vectors": self.count,
            "dimension": self.dim,
            "total_documents": len(self.id_to_document),
        }
