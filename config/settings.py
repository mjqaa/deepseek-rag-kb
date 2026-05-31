"""
配置管理模块
-----------
统一管理所有配置项，支持从环境变量 / .env 文件加载。
所有模块通过 get_settings() 获取全局唯一配置实例（单例模式）。

模型切换说明：
  LLM_PROVIDER 控制问答大模型，PREPROCESS_PROVIDER 控制文档预处理模型。

  ┌─────────────────┬──────────────────┬──────────────────┐
  │ LLM_PROVIDER     │ 问答模型          │ 预处理模型        │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ deepseek（默认） │ DeepSeek         │ 无（仅本地切片）   │
  │ qwen             │ 通义千问          │ 无（仅本地切片）   │
  │ hybrid           │ DeepSeek         │ 通义千问          │
  └─────────────────┴──────────────────┴──────────────────┘

  也可以单独设置 PREPROCESS_PROVIDER=qwen 配合 LLM_PROVIDER=deepseek 使用。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录：config/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 自动加载项目根目录下的 .env 文件（优先级高于系统环境变量）
# override=False: 命令行/系统环境变量优先级高于 .env，方便临时切换模式测试
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings:
    """
    全局配置类 — 单例模式

    所有配置项都可以通过环境变量覆盖，代码中提供合理默认值。
    扩展新配置时只需在此类中添加属性即可。
    """

    _instance = None  # 单例缓存

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load()

    # ========================================================================
    # 配置加载
    # ========================================================================

    def _load(self):
        """从环境变量加载所有配置项，缺失必填项时给出明确提示。"""

        # ---------- 模型提供商选择 ----------
        # deepseek : 问答用 DeepSeek
        # qwen     : 问答用千问
        # hybrid   : 预处理用千问 + 问答用 DeepSeek
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
        # 预处理提供商（可独立覆盖，优先级高于 LLM_PROVIDER=hybrid 的推断）
        self.PREPROCESS_PROVIDER = os.getenv("PREPROCESS_PROVIDER", "").lower()

        # ---------- DeepSeek API ----------
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_BASE_URL = os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

        # ---------- 通义千问 API ----------
        self.QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
        self.QWEN_BASE_URL = os.getenv(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
        # 预处理用的模型（通常用便宜的 turbo 即可）
        self.QWEN_PREPROCESS_MODEL = os.getenv("QWEN_PREPROCESS_MODEL", "qwen-turbo")

        # ---------- Embedding ----------
        self.LOCAL_EMBEDDING_MODEL = os.getenv(
            "LOCAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

        # ---------- 文档切片 ----------
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

        # ---------- 检索 ----------
        self.TOP_K = int(os.getenv("TOP_K", "5"))

        # ---------- HuggingFace 镜像（国内加速）----------
        hf_endpoint = os.getenv("HF_ENDPOINT", "")
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        # ---------- 存储路径 ----------
        kb_path = os.getenv("KB_STORAGE_PATH", "./kb_storage")
        self.KB_STORAGE_PATH = str(PROJECT_ROOT / kb_path)

        # ---------- 数据目录 ----------
        self.DATA_DIR = str(PROJECT_ROOT / "data" / "documents")

    # ========================================================================
    # 便捷判断属性
    # ========================================================================

    @property
    def use_qwen_for_chat(self) -> bool:
        """问答是否使用千问。"""
        return self.LLM_PROVIDER == "qwen"

    @property
    def use_qwen_for_preprocess(self) -> bool:
        """预处理是否使用千问。"""
        if self.PREPROCESS_PROVIDER == "qwen":
            return True
        if self.LLM_PROVIDER == "hybrid":
            return True
        return False

    @property
    def use_deepseek_for_chat(self) -> bool:
        """问答是否使用 DeepSeek。"""
        return self.LLM_PROVIDER in ("deepseek", "hybrid")

    # ========================================================================
    # 校验 & 工具方法
    # ========================================================================

    def validate_api_key(self) -> bool:
        """
        检查当前 LLM_PROVIDER 对应的 API Key 是否已配置。

        密钥校验策略（从不在日志中输出密钥明文）：
          仅检查是否为空 / 是否为示例占位符。
        """
        provider = self.LLM_PROVIDER
        if provider in ("deepseek", "hybrid"):
            key = self.DEEPSEEK_API_KEY
            if not key or key.startswith("sk-your-"):
                return False
        if provider in ("qwen", "hybrid"):
            key = self.QWEN_API_KEY
            if not key or key.startswith("sk-your-"):
                return False
        return True

    def validate_preprocess_key(self) -> bool:
        """检查预处理模型的 API Key 是否已配置。"""
        if not self.use_qwen_for_preprocess:
            return True  # 不需要预处理 API
        key = self.QWEN_API_KEY
        return bool(key and not key.startswith("sk-your-"))

    def ensure_dirs(self):
        """确保必要的目录存在（向量库存储目录、数据目录）。"""
        os.makedirs(self.KB_STORAGE_PATH, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

    def display(self) -> str:
        """返回当前配置的摘要信息（隐藏敏感字段）。"""
        lines = [
            f"LLM Provider : {self.LLM_PROVIDER}",
            f"Preprocess   : {'千问' if self.use_qwen_for_preprocess else '无（仅本地切片）'}",
        ]
        if self.use_deepseek_for_chat:
            lines.append(f"DeepSeek URL : {self.DEEPSEEK_BASE_URL}")
            lines.append(f"DeepSeek Model: {self.DEEPSEEK_CHAT_MODEL}")
        if self.use_qwen_for_chat or self.use_qwen_for_preprocess:
            lines.append(f"Qwen URL     : {self.QWEN_BASE_URL}")
            if self.use_qwen_for_chat:
                lines.append(f"Qwen Chat    : {self.QWEN_CHAT_MODEL}")
            if self.use_qwen_for_preprocess:
                lines.append(f"Qwen Preproc : {self.QWEN_PREPROCESS_MODEL}")
        lines.extend([
            f"Embed Model  : {self.LOCAL_EMBEDDING_MODEL}",
            f"Device       : {self.EMBEDDING_DEVICE}",
            f"Chunk Size   : {self.CHUNK_SIZE}",
            f"Chunk Overlap: {self.CHUNK_OVERLAP}",
            f"Top-K        : {self.TOP_K}",
            f"KB Storage   : {self.KB_STORAGE_PATH}",
            f"Data Dir     : {self.DATA_DIR}",
        ])
        return "\n".join(lines)


# ========================================================================
# 便捷获取函数（全局唯一入口）
# ========================================================================

def get_settings() -> Settings:
    """获取全局配置单例。"""
    return Settings()
