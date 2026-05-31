"""
工具函数模块
-----------
提供项目中复用的通用功能。
"""

import os
import time
from typing import List


def timing(func):
    """
    装饰器：打印函数执行耗时。

    用法：
        @timing
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [耗时] {func.__name__}: {elapsed:.2f} 秒")
        return result
    return wrapper


def format_file_size(size_bytes: int) -> str:
    """
    将字节数转换为可读的文件大小字符串。

    Args:
        size_bytes: 文件字节数

    Returns:
        如 "1.5 MB"
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def scan_documents(data_dir: str, extensions: List[str] = None) -> List[str]:
    """
    扫描指定目录下的文档文件。

    Args:
        data_dir: 数据目录路径
        extensions: 要扫描的文件扩展名列表，默认 ['.pdf', '.txt']

    Returns:
        文件绝对路径列表
    """
    if extensions is None:
        extensions = [".pdf", ".txt"]

    files = []
    if not os.path.isdir(data_dir):
        return files

    for fname in os.listdir(data_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in extensions:
            files.append(os.path.join(data_dir, fname))

    return sorted(files)


def print_banner():
    """打印启动横幅。"""
    print("""
╔══════════════════════════════════════════════════╗
║       DeepSeek RAG 本地私有知识库系统              ║
║       Powered by DeepSeek API + FAISS             ║
╚══════════════════════════════════════════════════╝
    """)


def print_help():
    """打印 CLI 帮助信息。"""
    help_text = """
══════════════════════════════════════════════════
  DeepSeek RAG 知识库 — 命令说明
══════════════════════════════════════════════════

  可用命令:
    build       从 data/documents/ 目录构建/更新知识库索引
    search      检索知识库（输入关键词，返回相关片段）
    chat        进入交互式 RAG 问答模式
    test        测试 DeepSeek API 连接
    stats       查看知识库索引状态
    help        显示本帮助信息
    quit/exit   退出程序

  快速开始:
    1. 将 PDF/TXT 文档放入 data/documents/ 目录
    2. 执行 build 构建索引
    3. 执行 chat 进入问答模式

══════════════════════════════════════════════════
    """
    print(help_text)
