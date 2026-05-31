"""
检索召回模块
-----------
负责将用户问题转化为向量，在 FAISS 索引中检索最相关的文档片段。

职责：
  1. 接收用户查询文本
  2. 调用 Embedding 模型将查询向量化
  3. 调用 FAISSVectorStore 进行相似度搜索
  4. 返回 Top-K 相关文档片段及相似度分数

扩展接口：
  如需混合检索（如 BM25 + 向量）、重排序（Reranker），
  可在本模块中扩展 Retriever 类。
"""

from typing import List, Tuple

from core.document_loader import Document
from core.embeddings import BaseEmbedding
from core.vector_store import FAISSVectorStore


class Retriever:
    """
    检索器。

    封装"查询 → 向量化 → FAISS搜索 → 返回结果"的完整检索链路。
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding: BaseEmbedding,
        top_k: int = 5,
    ):
        """
        Args:
            vector_store: FAISS 向量库实例
            embedding: Embedding 实例（用于将查询文本向量化）
            top_k: 默认返回的最相关文档数
        """
        self.vector_store = vector_store
        self.embedding = embedding
        self.top_k = top_k

    # ========================================================================
    # 公开接口
    # ========================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        根据查询文本检索最相关的文档片段。

        Args:
            query: 用户查询文本
            top_k: 返回条数，默认使用初始化时的值
            min_score: 最低相似度阈值（低于此分数的结果会被过滤）

        Returns:
            [(Document, score), ...] 按相似度降序排列
            score 为余弦相似度（-1 ~ 1）
        """
        # 1. 将查询文本转为向量
        query_vec = self.embedding.embed_query(query)

        # 2. FAISS 检索
        k = top_k if top_k is not None else self.top_k
        results = self.vector_store.search(query_vec, top_k=k)

        # 3. 过滤低分结果
        if min_score is not None:
            results = [(doc, s) for doc, s in results if s >= min_score]

        return results

    def retrieve_as_context(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
    ) -> str:
        """
        将检索结果拼接为一段上下文字符串，方便送入 LLM。

        格式：
          【来源1】文件名 (第X页)
          内容...
          （相似度: 0.923）

          【来源2】文件名
          内容...
          （相似度: 0.856）

        Args:
            query: 用户查询文本
            top_k: 返回条数
            min_score: 最低相似度阈值

        Returns:
            格式化后的上下文字符串
        """
        results = self.retrieve(query, top_k=top_k, min_score=min_score)

        if not results:
            return "（未找到相关文档）"

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "")
            page_info = f" (第{page}页)" if page else ""

            parts.append(
                f"【来源{i}】{source}{page_info}\n"
                f"{doc.content}\n"
                f"（相似度: {score:.4f}）"
            )

        return "\n\n".join(parts)

    def get_sources(self, results: List[Tuple[Document, float]]) -> List[dict]:
        """
        从检索结果中提取来源信息，方便溯源。

        Returns:
            [{"source": ..., "page": ..., "score": ...}, ...]
        """
        return [
            {
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page"),
                "chunk_idx": doc.metadata.get("chunk_idx"),
                "score": round(score, 4),
            }
            for doc, score in results
        ]
