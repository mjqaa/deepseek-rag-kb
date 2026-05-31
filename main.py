"""
DeepSeek RAG 本地私有知识库系统 — 控制台入口
============================================

完整的 RAG 知识库管线：
  文档加载 → [千问文本清洗] → 文本切片 → 向量化 → FAISS 存储 → 检索召回 → LLM 问答

支持的模型组合（通过 .env 中 LLM_PROVIDER 切换）：
  deepseek : 问答用 DeepSeek，无预处理
  qwen     : 问答用千问，无预处理
  hybrid   : 预处理用千问 + 问答用 DeepSeek（推荐多模型协同）

使用方法：
  python main.py                # 进入交互式命令行
  python main.py build          # 构建知识库索引
  python main.py ingest [dir]   # 批量导入+预处理+入库（新增）
  python main.py chat           # 直接进入问答模式
  python main.py test           # 测试 API 连接（自动检测当前提供商）
"""

import os
import sys

# 确保项目根目录在 Python 搜索路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from core.document_loader import get_loader, get_supported_extensions
from core.text_splitter import (
    RecursiveTextSplitter,
    QwenTextCleaner,
    preprocess_documents,
)
from core.embeddings import get_embedding
from core.vector_store import FAISSVectorStore
from core.retriever import Retriever
from core.llm_chain import LLMChat, QwenChat, get_llm_chat
from utils.helpers import scan_documents, timing, print_banner, print_help


class RAGPipeline:
    """
    RAG 管线编排器。

    将所有模块串联为完整的 RAG 流程，对外暴露以下核心操作：
      build        — 标准索引构建
      batch_ingest — 批量预处理 + 构建入库（支持千问文本清洗）
      search       — 知识库检索
      chat         — RAG 问答
      test         — API 连通性测试
      stats        — 索引状态

    模型选择由 .env 中的 LLM_PROVIDER 控制，无需修改代码。
    """

    def __init__(self):
        self.settings = get_settings()
        self.settings.ensure_dirs()

        # 延迟初始化（首用时才加载大模型，避免 import 时的等待）
        self._embedding = None
        self._vector_store = None
        self._retriever = None
        self._llm = None
        self._cleaner = None  # 千问文本清洗器（延迟加载）

    # ========================================================================
    # 属性（延迟加载）
    # ========================================================================

    @property
    def embedding(self):
        if self._embedding is None:
            print("\n[初始化] 加载 Embedding 模型...")
            self._embedding = get_embedding(embedding_type="local")
            print(f"  Embedding 维度: {self._embedding.dim}")
        return self._embedding

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = FAISSVectorStore(embedding=self.embedding)
        return self._vector_store

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = Retriever(
                vector_store=self.vector_store,
                embedding=self.embedding,
                top_k=self.settings.TOP_K,
            )
        return self._retriever

    @property
    def llm(self):
        """问答大模型实例（根据 LLM_PROVIDER 自动选择 DeepSeek 或千问）。"""
        if self._llm is None:
            self._llm = get_llm_chat(verbose=True)
        return self._llm

    @property
    def cleaner(self):
        """千问文本清洗器实例。"""
        if self._cleaner is None:
            self._cleaner = QwenTextCleaner()
        return self._cleaner

    def _show_model_info(self):
        """打印当前模型配置摘要。"""
        print()
        print(self.settings.display())
        print()

    # ========================================================================
    # 构建知识库索引（原有方法，内部新增预处理步骤）
    # ========================================================================

    def _load_and_chunk_documents(self, data_dir: str = None, use_preprocess: bool = None):
        """
        通用文档加载+预处理+切片管线。

        供 build_index() 和 batch_ingest() 共用，避免代码重复。

        Args:
            data_dir: 文档目录，默认使用配置中的 DATA_DIR
            use_preprocess: 是否启用千问文本清洗，默认根据配置自动判断

        Returns:
            (chunks, file_count, doc_count): 切片列表、文件数、原始文档段数
        """
        data_dir = data_dir or self.settings.DATA_DIR
        if use_preprocess is None:
            use_preprocess = self.settings.use_qwen_for_preprocess

        # Step 1: 扫描文档
        print(f"\n[Step 1/4] 扫描文档目录: {data_dir}")
        files = scan_documents(data_dir, get_supported_extensions())
        if not files:
            print("\n  [错误] 未找到任何文档文件！")
            print(f"  请将 PDF/TXT 文件放入: {data_dir}")
            print(f"  支持格式: {get_supported_extensions()}")
            return None, 0, 0

        print(f"  找到 {len(files)} 个文件:")
        for f in files:
            size = os.path.getsize(f)
            print(f"    - {os.path.basename(f)} ({size:,} bytes)")

        # Step 2: 加载文档
        print(f"\n[Step 2/4] 加载文档内容...")
        all_documents = []
        for file_path in files:
            try:
                loader = get_loader(file_path)
                docs = loader.load(file_path)
                all_documents.extend(docs)
                print(f"  [OK] {os.path.basename(file_path)} -> {len(docs)} 页/段")
            except Exception as e:
                print(f"  [FAIL] {os.path.basename(file_path)}: {e}")
                continue

        if not all_documents:
            print("\n  [错误] 所有文件加载失败，请检查文件是否损坏")
            return None, 0, 0

        file_count = len(files)
        doc_count = len(all_documents)
        print(f"  共加载 {doc_count} 个原始文档段")

        # Step 2.5: 文本预处理（千问清洗 或 本地基础清洗）
        all_documents = preprocess_documents(all_documents, use_qwen=use_preprocess)

        # Step 3: 文本切片
        print(f"\n[Step 3/4] 文本切片...")
        print(f"  切片大小: {self.settings.CHUNK_SIZE}  重叠: {self.settings.CHUNK_OVERLAP}")
        splitter = RecursiveTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split(all_documents)
        print(f"  切片完成: {doc_count} 段 → {len(chunks)} 个 Chunk")

        return chunks, file_count, doc_count

    @timing
    def build_index(self):
        """
        标准构建知识库索引。

        流程：
          1. 扫描 data/documents/ 目录下的 PDF/TXT 文件
          2. 逐个文件加载，提取文本
          3. [可选] 千问文本清洗（根据 LLM_PROVIDER 配置）
          4. 文本切片（chunk）
          5. 向量化（embedding）
          6. 存入 FAISS 索引并持久化到磁盘
        """
        print("\n" + "=" * 60)
        print("  知识库索引构建")
        print("=" * 60)
        self._show_model_info()

        chunks, file_count, doc_count = self._load_and_chunk_documents()
        if chunks is None:
            return False

        # Step 4: 向量化 + 构建 FAISS 索引
        print(f"\n[Step 4/4] 向量化并构建 FAISS 索引...")
        self.vector_store.build_from_documents(chunks)

        # 持久化保存
        save_path = self.settings.KB_STORAGE_PATH
        self.vector_store.save(save_path)

        print("\n" + "=" * 60)
        print(f"  [成功] 知识库构建完成！")
        print(f"  源文件    : {file_count} 个")
        print(f"  原始片段  : {doc_count} 段")
        print(f"  Chunk 数  : {self.vector_store.count}")
        print(f"  向量维度  : {self.vector_store.dim}")
        print(f"  存储位置  : {save_path}")
        print("=" * 60)
        return True

    # ========================================================================
    # 批量导入入库（新增方法）
    # ========================================================================

    @timing
    def batch_ingest(self, data_dir: str = None):
        """
        批量喂数据：一键完成「文档扫描 → 千问预处理 → 切片 → 向量化 → 入库」。

        与 build_index 的区别：
          - build_index: 使用基础流程，适合日常更新少量文档
          - batch_ingest: 专为大批量文档设计，详细展示每步进度和统计

        使用方法：
          python main.py ingest                    # 导入默认目录
          python main.py ingest /path/to/docs      # 导入指定目录

        Args:
            data_dir: 文档目录，默认使用配置中的 DATA_DIR
        """
        data_dir = data_dir or self.settings.DATA_DIR

        print("\n" + "=" * 60)
        print("  批量文档导入入库（Batch Ingest）")
        print("=" * 60)
        self._show_model_info()

        # 预处理方式确认
        use_preprocess = self.settings.use_qwen_for_preprocess
        if use_preprocess:
            preprocess_desc = "千问 API 智能清洗"
            if not self.settings.validate_preprocess_key():
                print("\n  [错误] 千问预处理模式需要配置 QWEN_API_KEY")
                print("  请在 .env 中添加: QWEN_API_KEY=your-key")
                return False
        else:
            preprocess_desc = "本地基础清洗（去首尾空白）"
        print(f"  预处理模式: {preprocess_desc}")

        # 复用通用文档加载+切片管线
        chunks, file_count, doc_count = self._load_and_chunk_documents(
            data_dir=data_dir,
            use_preprocess=use_preprocess,
        )
        if chunks is None:
            return False

        # 检查是否增量更新
        existing_count = self.vector_store.count if not self.vector_store.is_empty else 0
        if existing_count > 0:
            print(f"\n  [注意] 当前索引已有 {existing_count} 条向量，将覆盖重建")

        # 向量化 + 构建索引
        print(f"\n[Step 4/4] 向量化并构建 FAISS 索引...")
        self.vector_store.build_from_documents(chunks)

        # 持久化
        save_path = self.settings.KB_STORAGE_PATH
        self.vector_store.save(save_path)

        # 汇总报告
        print("\n" + "=" * 60)
        print("  批量导入入库完成")
        print("=" * 60)
        print(f"  源文件    : {file_count} 个")
        print(f"  原始片段  : {doc_count} 段")
        print(f"  入库 Chunk: {self.vector_store.count} 条")
        print(f"  向量维度  : {self.vector_store.dim}")
        print(f"  预处理    : {preprocess_desc}")
        print(f"  存储位置  : {save_path}")
        print("=" * 60)
        return True

    # ========================================================================
    # 加载已有索引
    # ========================================================================

    def load_index(self) -> bool:
        """
        加载已存在的 FAISS 索引。

        Returns:
            True 加载成功，False 索引不存在
        """
        return self.vector_store.load(self.settings.KB_STORAGE_PATH)

    # ========================================================================
    # 检索
    # ========================================================================

    def search(self, query: str):
        """
        检索知识库并展示结果。

        Args:
            query: 查询文本
        """
        if self.vector_store.is_empty:
            print("\n  [提示] 知识库为空，请先执行 build 构建索引")
            return

        print(f"\n  检索: \"{query}\"")
        print("-" * 40)

        context = self.retriever.retrieve_as_context(query)
        print(context)

        # 额外展示来源信息
        results = self.retriever.retrieve(query)
        if results:
            print("\n  [来源统计]")
            for src in self.retriever.get_sources(results):
                print(f"    {src['source']}  page={src['page']}  score={src['score']}")

    # ========================================================================
    # RAG 问答
    # ========================================================================

    def chat_once(self, question: str, stream: bool = True):
        """
        单轮 RAG 问答。

        Args:
            question: 用户问题
            stream: 是否流式输出
        """
        if self.vector_store.is_empty:
            print("\n  [提示] 知识库为空，请先执行 build 构建索引\n")
            return

        # 1. 检索相关文档
        context = self.retriever.retrieve_as_context(question)

        if context == "（未找到相关文档）":
            print("\n  [提示] 知识库中未找到与问题相关的文档片段")
            return

        # 2. 展示检索到的来源
        results = self.retriever.retrieve(question)
        print("\n" + "-" * 50)
        for src in self.retriever.get_sources(results):
            print(f"  [来源] {src['source']}  page={src['page']}  score={src['score']}")
        print("-" * 50)

        # 3. LLM 生成回答
        print("\n  [AI] 回答:", end="")
        answer = self.llm.rag_chat(question, context, stream=stream)
        # 非流式模式下，rag_chat 返回完整回答字符串，需要手动打印
        if not stream:
            print(answer)
            print()

    # ========================================================================
    # 交互式问答循环
    # ========================================================================

    def interactive_chat(self):
        """交互式 RAG 问答模式（持续对话直到用户退出）。"""
        if self.vector_store.is_empty:
            print("\n  [提示] 知识库为空，请先执行 build 构建索引\n")
            return

        print("\n" + "=" * 60)
        print("  交互式 RAG 问答模式")
        print("  输入问题后回车，输入 quit 退出")
        print("=" * 60)

        round_num = 0
        while True:
            try:
                question = input(f"\n  [Q{round_num + 1}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  再见！")
                break

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("  再见！")
                break

            round_num += 1
            self.chat_once(question, stream=True)

    # ========================================================================
    # API 连通性测试（升级为多模型感知）
    # ========================================================================

    def test_api(self):
        """测试当前配置的 LLM 提供商 API 连接。"""
        settings = self.settings

        print("\n" + "=" * 60)
        print("  API 连通性测试")
        print("=" * 60)
        print(f"  LLM_PROVIDER: {settings.LLM_PROVIDER}")
        print(f"  问答模型: {'千问' if settings.use_qwen_for_chat else 'DeepSeek'}")
        if settings.use_qwen_for_preprocess:
            print(f"  预处理模型: 千问")
        print()

        all_ok = True

        # 测试问答模型
        if settings.use_deepseek_for_chat:
            if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY.startswith("sk-your-"):
                print("  [失败] DeepSeek API Key 未配置，跳过测试")
                all_ok = False
            else:
                print("--- DeepSeek 问答模型 ---")
                ds = LLMChat()
                if not ds.test_connection():
                    all_ok = False
                print()

        if settings.use_qwen_for_chat:
            if not settings.QWEN_API_KEY or settings.QWEN_API_KEY.startswith("sk-your-"):
                print("  [失败] 千问 API Key 未配置，跳过测试")
                all_ok = False
            else:
                print("--- 千问 问答模型 ---")
                qw = QwenChat()
                if not qw.test_connection():
                    all_ok = False
                print()

        if settings.use_qwen_for_preprocess and not settings.use_qwen_for_chat:
            # hybrid 模式下单独测试千问预处理
            if not settings.QWEN_API_KEY or settings.QWEN_API_KEY.startswith("sk-your-"):
                print("  [失败] 千问 API Key 未配置（预处理需要），跳过测试")
                all_ok = False
            else:
                print("--- 千问 预处理模型 ---")
                qw = QwenChat(model=settings.QWEN_PREPROCESS_MODEL)
                if not qw.test_connection():
                    all_ok = False
                print()

        if all_ok:
            print("  [成功] 所有 API 连接正常")
        else:
            print("  [警告] 部分 API 连接失败，请检查配置")

    # ========================================================================
    # 索引状态
    # ========================================================================

    def show_stats(self):
        """显示知识库索引状态。"""
        print("\n" + "=" * 60)
        print("  知识库索引状态")
        print("=" * 60)
        self._show_model_info()

        # 检查磁盘上的索引文件
        import os
        index_path = os.path.join(self.settings.KB_STORAGE_PATH, "index.faiss")
        meta_path = os.path.join(self.settings.KB_STORAGE_PATH, "metadata.pkl")

        print(f"\n  索引文件  : {'[存在]' if os.path.exists(index_path) else '[不存在]'} ({index_path})")
        print(f"  元数据    : {'[存在]' if os.path.exists(meta_path) else '[不存在]'} ({meta_path})")

        if self.vector_store.is_empty:
            # 尝试加载
            loaded = self.load_index()
            if not loaded:
                print("\n  [提示] 知识库尚未构建，请执行 build")
                return

        stats = self.vector_store.get_stats()
        print(f"\n  向量总数  : {stats['total_vectors']}")
        print(f"  向量维度  : {stats['dimension']}")
        print(f"  文档片段  : {stats['total_documents']}")

        # 列出数据目录中的源文件
        files = scan_documents(self.settings.DATA_DIR, get_supported_extensions())
        print(f"\n  源文档 ({len(files)} 个):")
        for f in files:
            size = os.path.getsize(f)
            print(f"    - {os.path.basename(f)} ({size:,} bytes)")


# ========================================================================
# CLI 命令路由
# ========================================================================

def print_extended_help():
    """打印扩展帮助信息（包含新增命令）。"""
    help_text = """
══════════════════════════════════════════════════
  DeepSeek RAG 知识库 — 命令说明
══════════════════════════════════════════════════

  核心命令:
    build       从 data/documents/ 目录构建/更新知识库索引
    ingest      批量导入：扫描文档 + 千问预处理 + 切片 + 入库
                （可选指定目录: ingest /path/to/docs）
    chat        进入交互式 RAG 问答模式
    search      检索知识库（输入关键词，返回相关片段）

  辅助命令:
    test        测试 API 连接（自动检测 DeepSeek/千问）
    stats       查看知识库索引状态和当前模型配置
    help        显示本帮助信息
    quit/exit   退出程序

  模型切换:
    修改 .env 中的 LLM_PROVIDER 即可切换:
      deepseek → DeepSeek 问答
      qwen     → 千问 问答
      hybrid   → 千问预处理 + DeepSeek 问答（推荐）

  快速开始:
    1. 配置 .env（API Key + LLM_PROVIDER）
    2. 将 PDF/TXT 文档放入 data/documents/ 目录
    3. 执行 ingest 或 build 构建索引
    4. 执行 chat 进入问答模式

══════════════════════════════════════════════════
    """
    print(help_text)


def main():
    """
    程序入口。

    支持两种运行模式：
      1. 命令行参数模式: python main.py build|ingest|chat|search|test|stats
      2. 交互式菜单模式: python main.py（无参数时进入）
    """
    print_banner()

    # 初始化管线
    pipeline = RAGPipeline()

    # 展示当前模型配置（始终显示，方便用户确认）
    settings = get_settings()
    print(f"  模型配置: LLM_PROVIDER={settings.LLM_PROVIDER}", end="")
    if settings.use_qwen_for_chat:
        print(f" (千问 {settings.QWEN_CHAT_MODEL})")
    elif settings.LLM_PROVIDER == "hybrid":
        print(f" (问答:DeepSeek + 预处理:千问)")
    else:
        print(f" (DeepSeek {settings.DEEPSEEK_CHAT_MODEL})")

    # 获取命令行参数
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else None

    # ------------------------------------------------------------------
    # 命令行参数模式
    # ------------------------------------------------------------------
    if cmd:
        if cmd == "build":
            pipeline.build_index()
        elif cmd == "ingest":
            # 支持可选目录参数: python main.py ingest /path/to/docs
            custom_dir = sys.argv[2] if len(sys.argv) > 2 else None
            pipeline.batch_ingest(data_dir=custom_dir)
        elif cmd == "chat":
            if pipeline.vector_store.is_empty:
                loaded = pipeline.load_index()
                if not loaded:
                    print("\n  [提示] 未找到已有索引，请先执行 build 或 ingest")
                    return
            pipeline.interactive_chat()
        elif cmd == "search":
            if pipeline.vector_store.is_empty:
                loaded = pipeline.load_index()
                if not loaded:
                    print("\n  [提示] 未找到已有索引，请先执行 build 或 ingest")
                    return
            query = input("\n  请输入搜索关键词: ").strip()
            if query:
                pipeline.search(query)
        elif cmd == "test":
            pipeline.test_api()
        elif cmd == "stats":
            pipeline.show_stats()
        elif cmd in ("help", "-h", "--help"):
            print_extended_help()
        else:
            print(f"\n  未知命令: {cmd}")
            print_extended_help()
        return

    # ------------------------------------------------------------------
    # 交互式菜单模式
    # ------------------------------------------------------------------
    # 先检查 API Key
    if not pipeline.settings.validate_api_key():
        print("\n  [警告] 未检测到有效的 API Key！")
        print(f"  当前 LLM_PROVIDER={settings.LLM_PROVIDER}")
        print("  请检查 .env 中对应的 API Key 是否已正确配置")
        print()

    # 尝试加载已有索引
    index_loaded = pipeline.load_index()
    if index_loaded:
        print(f"\n  [信息] 已加载知识库索引: {pipeline.vector_store.count} 条向量")
    else:
        print(f"\n  [提示] 知识库尚未构建，请将文档放入 data/documents/ 后执行 build 或 ingest")
        print(f"     文档目录: {pipeline.settings.DATA_DIR}")

    print_extended_help()

    # 命令循环
    while True:
        try:
            user_input = input("  [>>] 请输入命令 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not user_input:
            continue

        if user_input in ("quit", "exit", "q"):
            print("  再见！")
            break
        elif user_input == "build":
            pipeline.build_index()
        elif user_input == "ingest":
            pipeline.batch_ingest()
        elif user_input == "chat":
            if pipeline.vector_store.is_empty:
                loaded = pipeline.load_index()
                if not loaded:
                    print("\n  [提示] 未找到已有索引，请先执行 build 或 ingest")
                    continue
            pipeline.interactive_chat()
        elif user_input == "search":
            if pipeline.vector_store.is_empty:
                loaded = pipeline.load_index()
                if not loaded:
                    print("\n  [提示] 未找到已有索引，请先执行 build 或 ingest")
                    continue
            query = input("  请输入搜索关键词: ").strip()
            if query:
                pipeline.search(query)
        elif user_input == "test":
            pipeline.test_api()
        elif user_input == "stats":
            pipeline.show_stats()
        elif user_input in ("help", "h"):
            print_extended_help()
        else:
            print(f"  未知命令: {user_input}，输入 help 查看可用命令")


if __name__ == "__main__":
    main()
