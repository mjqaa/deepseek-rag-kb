"""
文档加载模块
-----------
支持 PDF（.pdf）和纯文本（.txt）两种格式。
设计为"策略模式"：每种格式一个 Loader，通过工厂函数 get_loader() 自动匹配。

扩展接口：
  新增文件格式时，只需：
  1. 新建一个继承 BaseLoader 的类，实现 load() 方法
  2. 在 get_loader() 中注册新的扩展名映射即可
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from pypdf import PdfReader


# ========================================================================
# 统一文档数据结构
# ========================================================================

class Document:
    """
    标准化文档对象，贯穿整个 RAG 管线。

    Attributes:
        content  (str)           : 文档文本内容
        metadata (dict)          : 元信息（来源文件名、路径、页码等）
    """

    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        src = self.metadata.get("source", "unknown")
        return f"Document(source={src}, len={len(self.content)})"


# ========================================================================
# 抽象基类（预留扩展接口）
# ========================================================================

class BaseLoader(ABC):
    """
    文档加载器抽象基类。
    所有文件格式加载器必须继承此类并实现 load() 方法。
    """

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        加载文件并返回 Document 列表。

        Args:
            file_path: 文件的绝对路径

        Returns:
            List[Document]: 文档对象列表（一个文件可能产生多个 Document，如 PDF 每页一个）
        """
        ...


# ========================================================================
# PDF 加载器
# ========================================================================

class PDFLoader(BaseLoader):
    """
    PDF 文档加载器。

    使用 pypdf 逐页提取文本，每页生成一个 Document 对象。
    这样可以保留页码信息，方便后续溯源。
    """

    def load(self, file_path: str) -> List[Document]:
        documents = []
        file_name = os.path.basename(file_path)

        try:
            reader = PdfReader(file_path)
        except Exception as e:
            raise RuntimeError(f"无法打开 PDF 文件 {file_path}: {e}")

        total_pages = len(reader.pages)

        for page_num in range(total_pages):
            try:
                page = reader.pages[page_num]
                text = page.extract_text()

                # 跳过空白页
                if not text or not text.strip():
                    continue

                documents.append(Document(
                    content=text.strip(),
                    metadata={
                        "source": file_name,
                        "file_path": file_path,
                        "page": page_num + 1,        # 页码从 1 开始
                        "total_pages": total_pages,
                        "file_type": "pdf",
                    }
                ))
            except Exception as e:
                # 单页解析失败不中断整体流程，打印警告继续
                print(f"  [警告] PDF 第 {page_num + 1} 页解析失败: {e}")
                continue

        return documents


# ========================================================================
# TXT 加载器
# ========================================================================

class TXTLoader(BaseLoader):
    """
    纯文本文件加载器。

    自动尝试多种编码（UTF-8 → GBK → Latin-1），
    避免因编码问题导致中文乱码。
    """

    # 按常见程度排列编码探测顺序
    _ENCODINGS = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]

    def load(self, file_path: str) -> List[Document]:
        file_name = os.path.basename(file_path)
        content = None
        used_encoding = None

        # 依次尝试不同编码读取
        for enc in self._ENCODINGS:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                used_encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise RuntimeError(
                f"无法识别文件编码，已尝试: {self._ENCODINGS}\n"
                f"文件: {file_path}"
            )

        if not content.strip():
            print(f"  [提示] 文件内容为空: {file_name}")
            return []

        return [Document(
            content=content.strip(),
            metadata={
                "source": file_name,
                "file_path": file_path,
                "encoding": used_encoding,
                "file_type": "txt",
            }
        )]


# ========================================================================
# Loader 工厂 — 根据扩展名自动匹配
# ========================================================================

# 扩展名 → Loader 类 映射表（扩展新格式时在这里注册即可）
_LOADER_REGISTRY: Dict[str, type] = {
    ".pdf": PDFLoader,
    ".txt": TXTLoader,
}


def get_loader(file_path: str) -> BaseLoader:
    """
    根据文件扩展名返回对应的 Loader 实例。

    Args:
        file_path: 文件路径

    Returns:
        BaseLoader 子类实例

    Raises:
        ValueError: 不支持的文件格式
    """
    ext = Path(file_path).suffix.lower()
    loader_cls = _LOADER_REGISTRY.get(ext)
    if loader_cls is None:
        supported = ", ".join(_LOADER_REGISTRY.keys())
        raise ValueError(f"不支持的文件格式 '{ext}'，当前支持: {supported}")
    return loader_cls()


def get_supported_extensions() -> List[str]:
    """返回所有支持的文件扩展名列表。"""
    return list(_LOADER_REGISTRY.keys())
