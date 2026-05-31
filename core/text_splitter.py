"""
文本切片模块
-----------
将长文档切分为适合 Embedding 模型处理的短片段。
采用"递归字符分割"策略：优先按段落切，段落过长再按句子，最后按字符。

核心参数：
  - chunk_size : 每个切片的期望最大长度（字符数）
  - chunk_overlap : 相邻切片之间的重叠长度（避免语义在边界断裂）

扩展接口：
  如需不同的切片策略（如语义切片、固定Token切片），
  只需实现新的 Splitter 类，保持 split() 接口一致即可。
"""

import re
from typing import List, Tuple

from config.settings import get_settings
from core.document_loader import Document


class RecursiveTextSplitter:
    """
    递归文本切片器（中文优化版）

    工作流程：
      1. 先按段落分隔符（\\n\\n）拆分
      2. 对于过长的段落，按句子分隔符（。！？\\n）继续拆分
      3. 句子仍过长时，按逗号/分号等次级分隔符拆分
      4. 兜底：按固定长度硬切

    切片之间保留 overlap 长度的重叠内容，
    确保语义不会在 chunk 边界处被截断。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Args:
            chunk_size: 每个 chunk 的最大字符数（中文一个字 ≈ 1~2 tokens）
            chunk_overlap: 相邻 chunk 重叠字符数
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 分层分隔符（从粗到细）
        self._separators = [
            "\n\n",     # 第一优先级：段落
            "\n",       # 第二优先级：换行
            "。",       # 第三优先级：句号
            "！",       # 感叹号
            "？",       # 问号
            "；",       # 分号
            "，",       # 逗号
            " ",        # 空格（英文分词边界）
            "",         # 兜底：逐字符切
        ]

    # ========================================================================
    # 公开接口
    # ========================================================================

    def split(self, documents: List[Document]) -> List[Document]:
        """
        对一批 Document 进行切片。

        Args:
            documents: 原始 Document 列表

        Returns:
            切片后的新 Document 列表，每个 chunk 保留原始 metadata 并追加 chunk 编号
        """
        chunks = []
        for doc in documents:
            chunks.extend(self._split_one(doc))
        return chunks

    # ========================================================================
    # 内部实现
    # ========================================================================

    def _split_one(self, document: Document) -> List[Document]:
        """对单个 Document 进行切片。"""
        text = document.content
        if len(text) <= self.chunk_size:
            # 短文直接作为一个 chunk
            return [self._make_chunk(text, document.metadata, chunk_idx=0)]

        # 递归拆分
        segments = self._recursive_split(text)
        return [
            self._make_chunk(seg, document.metadata, chunk_idx=i)
            for i, seg in enumerate(segments)
        ]

    def _recursive_split(self, text: str, sep_idx: int = 0) -> List[str]:
        """
        递归拆分核心逻辑。

        每一层使用对应的分隔符拆分文本；
        如果某一段仍然超过 chunk_size，则用下一级分隔符继续拆。
        """
        # 短文本直接返回
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 所有分隔符都用完了，强制按长度切
        if sep_idx >= len(self._separators):
            return self._force_split(text)

        separator = self._separators[sep_idx]

        # 兜底：空字符串表示逐字符切
        if separator == "":
            return self._force_split(text)

        # 按当前分隔符拆分
        parts = text.split(separator)

        # 如果分隔符没起到作用（整个文本就是一段），
        # 则降到下一级分隔符
        if len(parts) == 1:
            return self._recursive_split(text, sep_idx + 1)

        # 合并短片段（控制 chunk 数量），然后递归处理仍然过长的片段
        chunks: List[str] = []
        current = ""

        for part in parts:
            # 把分隔符加回去（除了空格这类）
            piece = part if separator in (" ", "") else part + separator

            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current.strip():
                    # 当前积累的片段作为一个 chunk，但检查是否真的合格
                    if len(current) <= self.chunk_size:
                        chunks.append(current)
                    else:
                        # 仍然过长，递归降级拆分
                        chunks.extend(self._recursive_split(current, sep_idx + 1))
                current = piece

        # 处理最后一段
        if current.strip():
            if len(current) <= self.chunk_size:
                chunks.append(current)
            else:
                chunks.extend(self._recursive_split(current, sep_idx + 1))

        # 添加 overlap（在相邻 chunk 之间拼接重叠内容）
        return self._apply_overlap(chunks)

    def _force_split(self, text: str) -> List[str]:
        """硬切：按 chunk_size 等长切割，overlap 在 _apply_overlap 中处理。"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        为相邻 chunk 添加重叠内容。

        做法：每个 chunk（除了首个）在前面拼接前一个 chunk 的末尾 overlap 长度文本。
        这样语义不会在边界处断裂，检索时能更准确地匹配上下文。
        """
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            # 取前一个 chunk 的末尾作为当前 chunk 的前缀
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            result.append(overlap_text + chunks[i])

        return result

    def _make_chunk(
        self, text: str, meta: dict, chunk_idx: int
    ) -> Document:
        """构建切片 Document 对象，保留原始 metadata 并补充 chunk 信息。"""
        new_meta = {**meta, "chunk_idx": chunk_idx}
        return Document(content=text.strip(), metadata=new_meta)


# ========================================================================
# QwenTextCleaner — 千问文本清洗预处理器（新增扩展）
# ========================================================================

# 文本清洗专用提示词
CLEANING_PROMPT = """你是一个专业的文档清洗助手。请对以下文本进行清理和规整，要求：

1. **去乱码**：删除无法识别的乱码字符、控制字符、重复无意义符号
2. **去冗余**：合并重复段落，删除无意义的空行和纯标点行
3. **排版规整**：统一中英文之间的空格，修正段落缩进，规范标点符号
4. **内容纠错**：修正明显的 OCR 错误、形近字错误、数字错误（不要改变原意）
5. **保留结构**：保留原文的层级结构和编号列表
6. **只输出清洗后的文本**，不要添加任何解释、说明或前缀

原始文本：
"""


class QwenTextCleaner:
    """
    基于千问 API 的文本清洗预处理器。

    在文档切片入库前，自动调用千问完成：
      - 去乱码、去冗余
      - 排版规整
      - 内容纠错（OCR 错误、形近字等）

    这样入库的文本质量更高，检索和问答效果更好。

    使用方式：
      cleaner = QwenTextCleaner()
      cleaned_text = cleaner.clean(raw_text)

      # 也可在 pipeline 预处理环节批量调用
      for doc in documents:
          doc.content = cleaner.clean(doc.content)
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        """
        Args:
            api_key: 千问 API Key，默认从配置读取
            base_url: API 地址，默认从配置读取
            model: 模型名，默认使用 QWEN_PREPROCESS_MODEL（推荐 qwen-turbo）
        """
        from openai import OpenAI

        settings = get_settings()

        self.api_key = api_key or settings.QWEN_API_KEY
        self.base_url = base_url or settings.QWEN_BASE_URL
        self.model = model or settings.QWEN_PREPROCESS_MODEL

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def clean(self, text: str) -> str:
        """
        清洗单段文本。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本；如果 API 调用失败则原样返回（保证流程不中断）
        """
        # 文本太短没必要清洗，直接返回
        if len(text.strip()) < 50:
            return text.strip()

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": CLEANING_PROMPT + text},
                ],
                temperature=0.1,   # 极低温度，确保输出稳定
                max_tokens=4096,   # 预处理允许更长输出
            )
            cleaned = response.choices[0].message.content.strip()
            return cleaned if cleaned else text.strip()
        except Exception as e:
            # 清洗失败时原样返回，不中断整体建库流程
            print(f"    [警告] 千问文本清洗失败: {e}，将使用原文")
            return text.strip()

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        批量清洗 Document 列表。

        Args:
            documents: 原始 Document 列表

        Returns:
            清洗后的新 Document 列表（metadata 保留不变，content 被清洗）
        """
        cleaned_docs = []
        for i, doc in enumerate(documents):
            original_len = len(doc.content)
            cleaned_content = self.clean(doc.content)
            cleaned_len = len(cleaned_content)
            print(f"    [{i+1}/{len(documents)}] 清洗: "
                  f"{doc.metadata.get('source', '?')} "
                  f"({original_len} → {cleaned_len} 字符)")
            cleaned_docs.append(Document(
                content=cleaned_content,
                metadata=doc.metadata,
            ))
        return cleaned_docs


# ========================================================================
# 预处理管线辅助函数
# ========================================================================

def preprocess_documents(
    documents: List[Document],
    use_qwen: bool = False,
) -> List[Document]:
    """
    文档预处理统一入口。

    根据配置决定是否调用千问清洗，预留未来扩展更多预处理步骤。

    Args:
        documents: 原始 Document 列表
        use_qwen: 是否启用千问文本清洗

    Returns:
        预处理后的 Document 列表
    """
    if use_qwen:
        print(f"\n  [预处理] 启用千问文本清洗（{len(documents)} 段）...")
        from config.settings import get_settings
        settings = get_settings()
        cleaner = QwenTextCleaner(model=settings.QWEN_PREPROCESS_MODEL)
        documents = cleaner.clean_documents(documents)
    else:
        print(f"\n  [预处理] 使用本地基础清洗（去首尾空白）...")
        for doc in documents:
            doc.content = doc.content.strip()

    return documents
