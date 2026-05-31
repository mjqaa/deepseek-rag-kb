"""
大模型问答链路模块
-----------------
负责构建 Prompt 并调用大模型 API 生成回答。

支持的提供商：
  - DeepSeek : 通过 OpenAI 兼容接口调用 deepseek-chat / deepseek-reasoner
  - 千问 Qwen: 通过 DashScope 兼容接口调用 qwen-turbo / qwen-plus / qwen-max

RAG 问答流程：
  1. 用户提问
  2. Retriever 检索相关文档片段
  3. 将检索结果 + 用户问题拼装成 Prompt
  4. 调用 LLM 生成基于文档的答案
  5. 返回答案 + 引用来源

切换方式：修改 .env 中 LLM_PROVIDER 即可。
"""

from openai import OpenAI

from config.settings import get_settings


# ========================================================================
# 提示词模板（DeepSeek 和千问共用）
# ========================================================================

SYSTEM_PROMPT = """你是一个专业的本地知识库问答助手。

你的回答严格基于用户提供的参考文档内容。请遵守以下规则：
1. 如果参考文档中有明确答案，请直接引用并注明【来源X】
2. 如果参考文档仅提供部分相关信息，请说明"根据已有文档，……"并给出推断
3. 如果参考文档与问题完全无关，请如实回答"当前知识库中没有找到相关信息"
4. 回答时请保持简洁、准确、条理清晰
5. 不要编造文档中不存在的信息"""


def build_prompt(context: str, question: str) -> str:
    """
    构建发给 LLM 的完整提示词。

    Args:
        context: 检索到的参考文档片段（由 Retriever.retrieve_as_context() 生成）
        question: 用户的原始问题

    Returns:
        完整的用户消息文本
    """
    return f"""请根据以下参考文档回答用户问题。

## 参考文档
{context}

## 用户问题
{question}

## 回答
"""


# ========================================================================
# LLMChat — DeepSeek 实现（原有代码，完全保留不变）
# ========================================================================

class LLMChat:
    """
    DeepSeek 大模型问答封装。

    兼容 OpenAI SDK 接口，通过 base_url 指向 DeepSeek API。
    支持普通对话和流式输出两种模式。

    扩展接口：
      如需接入千问等其它大模型，只需修改 base_url / api_key / model 即可，
      OpenAI SDK 标准接口保持不变。
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        """
        Args:
            api_key: DeepSeek API 密钥，默认从配置读取
            base_url: API 地址，默认从配置读取
            model: 模型名称，默认从配置读取
        """
        settings = get_settings()

        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        self.base_url = base_url or settings.DEEPSEEK_BASE_URL
        self.model = model or settings.DEEPSEEK_CHAT_MODEL

        # 初始化 OpenAI 兼容客户端（指向 DeepSeek）
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # ========================================================================
    # RAG 问答（核心方法）
    # ========================================================================

    def rag_chat(
        self,
        question: str,
        context: str,
        stream: bool = False,
    ) -> str:
        """
        基于检索上下文进行 RAG 问答。

        内部流程：
          1. 构建 System Prompt + Context + Question
          2. 调用 DeepSeek Chat API
          3. 返回模型回答

        Args:
            question: 用户问题
            context: Retriever 检索到的参考文档文本
            stream: 是否流式输出（控制台实时打字效果）

        Returns:
            模型生成的回答文本
        """
        user_message = build_prompt(context, question)

        if stream:
            return self._chat_stream(user_message)
        else:
            return self._chat_sync(user_message)

    def _chat_sync(self, user_message: str) -> str:
        """同步调用 LLM（等待完整响应后返回）。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,  # RAG 场景用低温，减少幻觉
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[错误] LLM 调用失败: {e}"

    def _chat_stream(self, user_message: str) -> str:
        """
        流式调用 LLM（逐 token 实时输出到控制台）。

        兼顾用户体验和完整结果收集。
        """
        full_response = ""

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )

            print()  # 换行
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_response += delta.content
            print()  # 最终换行
            print()

            return full_response.strip()

        except Exception as e:
            return f"[错误] LLM 流式调用失败: {e}"

    # ========================================================================
    # 普通对话（不基于知识库，用于测试连接）
    # ========================================================================

    def simple_chat(self, message: str) -> str:
        """
        普通对话（不检索知识库，直接问 LLM）。

        用途：测试 API 连接是否正常、验证 API Key 是否有效。

        Args:
            message: 用户消息

        Returns:
            LLM 回复文本
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[错误] API 调用失败: {e}"

    # ========================================================================
    # API 连通性测试
    # ========================================================================

    def test_connection(self) -> bool:
        """
        测试 API 连通性。

        Returns:
            True 表示连接正常，False 表示失败
        """
        print("  正在测试 DeepSeek API 连接...")
        try:
            reply = self.simple_chat("你好，请用中文回复'连接成功'")
            print(f"  API 响应: {reply}")
            return True
        except Exception as e:
            print(f"  API 连接失败: {e}")
            return False


# ========================================================================
# QwenChat — 通义千问实现（新增，接口与 LLMChat 完全一致）
# ========================================================================

class QwenChat:
    """
    通义千问大模型问答封装。

    通过 DashScope 兼容 OpenAI 接口调用千问模型。
    接口与 LLMChat 完全一致，可在 RAG 管线中无缝替换。

    支持的模型：
      - qwen-turbo  : 速度快、成本低，适合简单任务
      - qwen-plus   : 平衡性能与成本（推荐）
      - qwen-max    : 最强能力，适合复杂推理
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        """
        Args:
            api_key: 千问 API 密钥，默认从配置读取（QWEN_API_KEY）
            base_url: API 地址，默认从配置读取
            model: 模型名称，默认从配置读取（QWEN_CHAT_MODEL）
        """
        settings = get_settings()

        self.api_key = api_key or settings.QWEN_API_KEY
        self.base_url = base_url or settings.QWEN_BASE_URL
        self.model = model or settings.QWEN_CHAT_MODEL

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # ========================================================================
    # RAG 问答（核心方法，接口与 LLMChat 完全一致）
    # ========================================================================

    def rag_chat(
        self,
        question: str,
        context: str,
        stream: bool = False,
    ) -> str:
        """
        基于检索上下文进行 RAG 问答。

        Args:
            question: 用户问题
            context: Retriever 检索到的参考文档文本
            stream: 是否流式输出

        Returns:
            模型生成的回答文本
        """
        user_message = build_prompt(context, question)

        if stream:
            return self._chat_stream(user_message)
        else:
            return self._chat_sync(user_message)

    def _chat_sync(self, user_message: str) -> str:
        """同步调用千问（等待完整响应后返回）。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[错误] 千问调用失败: {e}"

    def _chat_stream(self, user_message: str) -> str:
        """流式调用千问（逐 token 实时输出到控制台）。"""
        full_response = ""

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )

            print()
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_response += delta.content
            print()
            print()

            return full_response.strip()

        except Exception as e:
            return f"[错误] 千问流式调用失败: {e}"

    # ========================================================================
    # 普通对话（用于测试连接）
    # ========================================================================

    def simple_chat(self, message: str) -> str:
        """普通对话，用于测试千问 API 连通性。"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[错误] 千问 API 调用失败: {e}"

    # ========================================================================
    # API 连通性测试
    # ========================================================================

    def test_connection(self) -> bool:
        """测试千问 API 连通性。"""
        print("  正在测试千问 API 连接...")
        try:
            reply = self.simple_chat("你好，请用中文回复'连接成功'")
            print(f"  千问 API 响应: {reply}")
            return True
        except Exception as e:
            print(f"  千问 API 连接失败: {e}")
            return False


# ========================================================================
# 工厂函数 — 根据配置自动选择模型
# ========================================================================

def get_llm_chat(verbose: bool = False) -> LLMChat:
    """
    根据 LLM_PROVIDER 配置返回对应的大模型实例。

    返回值类型：
      - LLM_PROVIDER=deepseek → LLMChat（DeepSeek）
      - LLM_PROVIDER=qwen     → QwenChat（千问）
      - LLM_PROVIDER=hybrid   → LLMChat（问答用 DeepSeek，预处理用千问）

    Args:
        verbose: 是否打印当前使用的模型名称（首次加载时由调用方决定）

    Returns:
        LLMChat 或 QwenChat 实例（接口完全一致，可互换使用）
    """
    settings = get_settings()

    if settings.use_qwen_for_chat:
        if verbose:
            print("  [模型] 问答使用: 通义千问")
        return QwenChat()
    else:
        if verbose:
            print("  [模型] 问答使用: DeepSeek")
        return LLMChat()
