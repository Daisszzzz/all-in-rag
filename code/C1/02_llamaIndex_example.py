import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载 .env 文件中的环境变量，例如 API_KEY
load_dotenv()


# ==================== 1. 配置大语言模型 LLM ====================

# 使用 AIHubMix 提供的 OpenAI 兼容接口
# OpenAILike 用于接入接口格式与 OpenAI API 类似的第三方模型服务
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",              # 指定使用的大语言模型
    api_key=os.getenv("DEEPSEEK_API_KEY"),   # 从环境变量中读取 API Key
    api_base="https://aihubmix.com/v1",      # AIHubMix 的 API 地址
    is_chat_model=True                       # 声明该模型属于 Chat Model
)

# 如果直接使用 DeepSeek 官方 API，可以使用下面的配置
# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )


# ==================== 2. 配置 Embedding 模型 ====================

# 使用 BGE 中文 Embedding 模型
# 作用：将文档 Chunk 和用户 Query 转换成向量，
# 从而通过向量相似度进行语义检索
Settings.embed_model = HuggingFaceEmbedding(
    "BAAI/bge-small-zh-v1.5"
)


# ==================== 3. 加载文档 ====================

# 使用 SimpleDirectoryReader 读取本地 Markdown 文件
# load_data() 会将文件内容加载为 LlamaIndex 的 Document 对象
docs = SimpleDirectoryReader(
    input_files=[
        "../../data/C1/markdown/easy-rl-chapter1.md"
    ]
).load_data()


# ==================== 4. 构建向量索引 ====================

# 根据 Document 构建 VectorStoreIndex
#
# 这一过程内部主要完成：
# Document
#   ↓
# 文本切分（Chunk / Node）
#   ↓
# Embedding 模型进行向量化
#   ↓
# 将文本节点及对应向量保存到 VectorStoreIndex
#
# 因此这里虽然没有显式写 TextSplitter，
# LlamaIndex 仍然会按照默认配置对文档进行切分
index = VectorStoreIndex.from_documents(docs)


# ==================== 5. 创建查询引擎 ====================

# 将 VectorStoreIndex 转换为 QueryEngine
#
# QueryEngine 会负责完整的 RAG 查询流程：
#
# 用户 Query
#   ↓
# Query Embedding
#   ↓
# 从 VectorStoreIndex 中检索相关 Chunk / Node
#   ↓
# 将检索结果组织成 Context
#   ↓
# 与 Prompt 一起发送给 LLM
#   ↓
# 生成最终回答
query_engine = index.as_query_engine()


# ==================== 6. 查看 Prompt ====================

# 查看 QueryEngine 内部使用的 Prompt 模板
# 可以帮助理解 LlamaIndex 最终是如何把：
#   用户问题 + 检索到的上下文
# 组织后发送给 LLM 的
print(query_engine.get_prompts())


# ==================== 7. 执行 RAG 查询 ====================

# 查询：“文中举了哪些例子？”
#
# LlamaIndex 会自动完成：
# 1. 将问题进行 Embedding
# 2. 检索与问题最相关的 Chunk / Node
# 3. 将检索结果作为 Context
# 4. 调用 LLM
# 5. 返回最终生成的回答
response = query_engine.query("文中举了哪些例子?")

print(response)