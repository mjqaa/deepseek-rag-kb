"""
DeepSeek RAG 知识库 — Web 可视化前端
=====================================
基于 Streamlit 的轻量 Web 界面，100% 复用项目现有 RAG 核心逻辑。

Embedding: 使用千问 API（DashScope），无需本地 torch 模型。
启动方式: streamlit run web_app.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config.settings import get_settings
from core.document_loader import Document
from core.embeddings import BaseEmbedding
from core.vector_store import FAISSVectorStore
from core.retriever import Retriever
from core.document_loader import get_loader
from core.text_splitter import RecursiveTextSplitter, preprocess_documents
from core.llm_chain import SYSTEM_PROMPT, build_prompt, LLMChat, QwenChat
from utils.helpers import scan_documents

st.set_page_config(
    page_title="DeepSeek RAG 知识库", page_icon="📚",
    layout="wide", initial_sidebar_state="expanded",
)

# ============================================================================
# 千问 API Embedding（替代本地模型，避免 torch/streamlit 冲突）
# ============================================================================

class QwenAPIEmbedding(BaseEmbedding):
    """基于千问 DashScope API 的文本向量化，OpenAI 兼容接口。"""

    def __init__(self, api_key=None, base_url=None, model="text-embedding-v1"):
        from openai import OpenAI
        settings = get_settings()
        self._api_key = api_key or settings.QWEN_API_KEY
        self._base_url = base_url or settings.QWEN_BASE_URL
        self._model = model
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        self._dim = 1536  # text-embedding-v1 输出 1536 维

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts):
        if not texts:
            return np.array([], dtype=np.float32)
        embeddings = []
        for text in texts:
            resp = self._client.embeddings.create(model=self._model, input=text)
            emb = np.array(resp.data[0].embedding, dtype=np.float32)
            # L2 归一化（兼容 FAISS IndexFlatIP 做余弦相似度）
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.stack(embeddings)

    def embed_query(self, text: str):
        return self.embed_texts([text])[0]


# ============================================================================
# 初始化
# ============================================================================

def bootstrap():
    if st.session_state.get("booted"):
        return True
    try:
        settings = get_settings()
        settings.ensure_dirs()
        embedding = QwenAPIEmbedding()
        vector_store = FAISSVectorStore(embedding=embedding)
        retriever = Retriever(vector_store=vector_store, embedding=embedding, top_k=settings.TOP_K)
        index_loaded = vector_store.load(settings.KB_STORAGE_PATH)

        st.session_state.settings = settings
        st.session_state.embedding = embedding
        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever
        st.session_state.index_loaded = index_loaded
        st.session_state.messages = []
        st.session_state.llm_provider = "deepseek"
        st.session_state.build_triggered = False
        st.session_state.need_rebuild = index_loaded  # 已有旧索引（384维），与API维度不同，需重建
        st.session_state.booted = True
        st.session_state.init_error = None
        return True
    except Exception as e:
        import traceback
        st.session_state.init_error = f"{e}\n\n{traceback.format_exc()}"
        st.session_state.booted = False
        return False


# ============================================================================
# 页面渲染
# ============================================================================

if not st.session_state.get("booted"):
    st.title("📚 DeepSeek RAG 知识库")
    if st.session_state.get("init_error"):
        st.error(f"初始化失败:\n\n```\n{st.session_state.init_error}\n```")
        if st.button("🔁 重试"):
            st.session_state.init_error = None
            st.rerun()
    else:
        with st.spinner("正在初始化千问 Embedding..."):
            ok = bootstrap()
        if ok:
            st.rerun()
        else:
            st.rerun()
    st.stop()

# ============================================================================
# 正常页面
# ============================================================================

S = st.session_state.settings
VS = st.session_state.vector_store
RET = st.session_state.retriever


def create_llm(provider: str):
    if provider == "qwen":
        return QwenChat()
    return LLMChat()


def stream_rag_response(llm, question: str, context: str):
    user_message = build_prompt(context, question)
    try:
        stream = llm._client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3, max_tokens=2048, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as e:
        yield f"\n\n> ⚠️ LLM 调用失败: {e}"


# ============================================================================
# 侧边栏
# ============================================================================

with st.sidebar:
    st.title("📚 RAG 知识库")
    st.caption("Embedding: 千问 API | DeepSeek + FAISS")
    st.divider()

    st.subheader("🧠 LLM 模式")
    provider_labels = {
        "deepseek": "DeepSeek 问答",
        "qwen": "通义千问 问答",
        "hybrid": "Hybrid（千问预处理 + DeepSeek 问答）",
    }
    provider = st.radio(
        "选择模型模式",
        options=["deepseek", "qwen", "hybrid"],
        format_func=lambda x: provider_labels[x],
        index=["deepseek", "qwen", "hybrid"].index(st.session_state.llm_provider),
        label_visibility="collapsed",
    )
    if provider != st.session_state.llm_provider:
        st.session_state.llm_provider = provider
        S.LLM_PROVIDER = provider
        st.rerun()

    if provider == "deepseek":
        st.caption(f"问答模型: DeepSeek ({S.DEEPSEEK_CHAT_MODEL})")
    elif provider == "qwen":
        st.caption(f"问答模型: 千问 ({S.QWEN_CHAT_MODEL})")
    else:
        st.caption(f"问答: DeepSeek  |  预处理: 千问")

    st.divider()

    st.subheader("📄 文档上传")
    uploaded_files = st.file_uploader(
        "上传 PDF / TXT", type=["pdf", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded_files:
        saved = 0
        for f in uploaded_files:
            target = os.path.join(S.DATA_DIR, f.name)
            if os.path.exists(target):
                st.warning(f"已存在，跳过: {f.name}")
                continue
            with open(target, "wb") as fout:
                fout.write(f.getbuffer())
            saved += 1
        if saved:
            st.success(f"已保存 {saved} 个文件")

    existing = scan_documents(S.DATA_DIR, [".pdf", ".txt"])
    if existing:
        with st.expander(f"📁 已有文档 ({len(existing)} 个)"):
            for fp in existing:
                fname = os.path.basename(fp)
                size = os.path.getsize(fp)
                st.caption(f"• {fname}  ({size:,} B)")

    st.divider()

    st.subheader("🔧 知识库构建")
    preprocess_hint = "千问 API 智能清洗" if provider == "hybrid" else "本地基础清洗"
    st.caption(f"预处理: {preprocess_hint}")

    # 提示：如果之前用本地模型（384维）建的库，需要用API（1536维）重建
    if VS.is_empty:
        st.caption("⚠️ 知识库为空，请上传文档后构建")

    if st.button("🔄 重新构建知识库", type="primary", use_container_width=True):
        st.session_state.build_triggered = True
        st.rerun()

    st.divider()

    st.subheader("📊 索引状态")
    if VS.is_empty:
        st.caption("向量数: 0（未构建）")
    else:
        st.caption(f"向量数: {VS.count}")
        st.caption(f"维度: {VS.dim}（千问 API）")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================================
# 知识库构建
# ============================================================================

if st.session_state.build_triggered:
    st.session_state.build_triggered = False
    with st.status("正在构建知识库...", expanded=True) as status:
        st.write("📂 扫描文档...")
        files = scan_documents(S.DATA_DIR, [".pdf", ".txt"])
        if not files:
            st.error("data/documents/ 目录为空，请先上传文档")
            status.update(label="构建失败", state="error")
            st.stop()
        st.write(f"  找到 {len(files)} 个文件")

        st.write("📖 加载文档...")
        all_docs = []
        for fp in files:
            try:
                loader = get_loader(fp)
                docs = loader.load(fp)
                all_docs.extend(docs)
                st.write(f"  ✓ {os.path.basename(fp)} → {len(docs)} 段")
            except Exception as e:
                st.write(f"  ✗ {os.path.basename(fp)}: {e}")
        if not all_docs:
            st.error("所有文件加载失败")
            status.update(label="构建失败", state="error")
            st.stop()

        use_qwen = provider == "hybrid"
        all_docs = preprocess_documents(all_docs, use_qwen=use_qwen)

        st.write("✂️ 文本切片...")
        splitter = RecursiveTextSplitter(chunk_size=S.CHUNK_SIZE, chunk_overlap=S.CHUNK_OVERLAP)
        chunks = splitter.split(all_docs)
        st.write(f"  {len(all_docs)} 段 → {len(chunks)} 个 Chunk")

        st.write("🧮 向量化（千问 API）& 构建 FAISS 索引...")
        VS.build_from_documents(chunks)
        VS.save(S.KB_STORAGE_PATH)

        status.update(label=f"构建完成 — {len(files)} 文件, {VS.count} 条向量", state="complete")
    st.success("知识库已就绪")
    time.sleep(0.5)
    st.rerun()


# ============================================================================
# 主区域
# ============================================================================

st.title("📚 DeepSeek RAG 知识库问答")
st.caption("千问 API Embedding + FAISS 向量库 + 大模型 RAG 检索增强生成")

if VS.is_empty:
    st.warning("⚠️ 知识库尚未构建。请在左侧边栏上传文档，然后点击「重新构建知识库」")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                for s in msg["sources"]:
                    line = f"**{s['source']}**"
                    if s.get("page"):
                        line += f" — 第{s['page']}页"
                    line += f" ｜ 相似度 `{s['score']:.4f}`"
                    st.caption(line)

prompt = st.chat_input(
    "请输入你的问题..." if not VS.is_empty else "请先构建知识库",
    disabled=VS.is_empty,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = RET.retrieve(prompt)
        sources = RET.get_sources(results)
        context = RET.retrieve_as_context(prompt)

        if context == "（未找到相关文档）":
            reply = "当前知识库中没有找到与问题相关的文档片段，请尝试更换提问方式。"
            st.info(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
        else:
            llm = create_llm(st.session_state.llm_provider)
            placeholder = st.empty()
            full = ""
            for token in stream_rag_response(llm, prompt, context):
                full += token
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)

            st.divider()
            st.caption("📎 **参考来源**")
            for s in sources:
                line = f"**{s['source']}**"
                if s.get("page"):
                    line += f" — 第{s['page']}页"
                line += f" ｜ 相似度 `{s['score']:.4f}`"
                st.caption(line)

            st.session_state.messages.append({"role": "assistant", "content": full, "sources": sources})
