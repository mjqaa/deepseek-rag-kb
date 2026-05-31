"""
Embedding 模型下载工具（国内专用）
===============================
使用 ModelScope（阿里云国内直连）下载，速度远快于 HuggingFace。

运行: python download_model.py
下载完成后运行: streamlit run web_app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache", "embedding_model")

if os.path.exists(LOCAL_DIR) and os.path.isdir(LOCAL_DIR):
    files = os.listdir(LOCAL_DIR)
    if len(files) > 5:
        print(f"[跳过] 模型已存在: {LOCAL_DIR}")
        print(f"  文件数: {len(files)}")
        sys.exit(0)

os.makedirs(os.path.dirname(LOCAL_DIR), exist_ok=True)
print(f"[开始] 下载 Embedding 模型...")
print(f"  模型: {MODEL_NAME}")
print(f"  目标: {LOCAL_DIR}")
print()

try:
    from modelscope import snapshot_download
    print("使用 ModelScope 下载（阿里云国内加速）...")
    result = snapshot_download(MODEL_NAME, cache_dir=LOCAL_DIR)
    print(f"\n[完成] 模型已保存到: {result}")
except ImportError:
    print("[提示] modelscope 未安装，使用 huggingface 镜像...")
    print("安装 modelscope 可获得更快的下载速度: pip install modelscope")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME, local_dir=LOCAL_DIR)
    print(f"\n[完成] 模型已保存到: {LOCAL_DIR}")
