from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. 配置与创建索引时相同的 Embedding 模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)


# 2. 指定之前保存的索引目录
persist_path = "./llamaindex_index_store"


# 3. 加载本地存储
storage_context = StorageContext.from_defaults(
    persist_dir=persist_path
)


# 4. 从存储中恢复 VectorStoreIndex
index = load_index_from_storage(
    storage_context
)

print("LlamaIndex 索引加载成功")


# 5. 创建 Retriever
retriever = index.as_retriever(
    similarity_top_k=3
)


# 6. 输入查询
query = "张三是谁？"


# 7. 执行相似性搜索
results = retriever.retrieve(query)


# 8. 输出检索结果
print(f"\n查询内容：{query}")
print("\n=== 相似度检索结果 ===")

for i, result in enumerate(results, start=1):
    print(f"\nTop {i}")
    print(f"相似度：{result.score}")
    print(f"文本：{result.node.get_content()}")