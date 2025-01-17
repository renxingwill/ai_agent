import os
from dotenv import load_dotenv
import json
import re
from llama_index.embeddings.openai import OpenAIEmbedding

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY1')
if not api_key:
    raise ValueError("环境变量 ZISHU_API_KEY1 未设置或为空，请检查 .env 文件！")    
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"
emb_model = "embedding-2"
from llama_index.llms.zhipuai import ZhipuAI
llm = ZhipuAI(
    api_key = api_key,
    model = chat_model,
)
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
embedding = ZhipuAIEmbedding(
    api_key = api_key,
    model = emb_model,
)


from llama_index.core import SimpleDirectoryReader,Document # 导入 LlamaIndex 库中的 SimpleDirectoryReader 和 Document 类，用于从目录中读取文件和表示文档
import os # 导入 Python 的 os 模块，用于与操作系统进行交互
print(f"当前工作目录: {os.getcwd()}") # 打印当前 Python 脚本的工作目录

documents = SimpleDirectoryReader(input_files=['./wow-agent/docs/问答手册.txt']).load_data() # 使用 SimpleDirectoryReader 读取指定路径下的文本文件，并将读取到的数据加载为 Document 对象列表

# 构建节点
from llama_index.core.node_parser import SentenceSplitter # 导入 LlamaIndex 库中的 SentenceSplitter 类，用于将文档分割成更小的语义单元（节点）
transformations = [SentenceSplitter(chunk_size = 512)] # 创建一个 SentenceSplitter 实例，设置每个 chunk 的大小为 512 个字符，并将其放入一个列表中作为转换步骤

from llama_index.core.ingestion.pipeline import run_transformations # 导入 LlamaIndex 库中的 run_transformations 函数，用于执行数据转换管道
nodes = run_transformations(documents, transformations=transformations) # 对加载的文档执行定义的转换步骤，将文档分割成节点列表

# 构建索引
from llama_index.vector_stores.faiss import FaissVectorStore # 导入 LlamaIndex 库中的 FaissVectorStore 类，用于使用 Faiss 库创建向量数据库
import faiss # 导入 Faiss 库，一个用于高效相似性搜索和密集向量聚类的库
from llama_index.core import StorageContext, VectorStoreIndex # 导入 LlamaIndex 库中的 StorageContext 和 VectorStoreIndex 类，用于管理存储上下文和创建向量索引

vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1024)) # 创建一个 FaissVectorStore 实例，使用 Faiss 的 IndexFlatL2 索引（L2 距离），向量维度设置为 1024
storage_context = StorageContext.from_defaults(vector_store=vector_store) # 从默认配置创建一个 StorageContext 实例，并将创建的向量数据库关联到该上下文

index = VectorStoreIndex(
    nodes = nodes, # 使用之前创建的节点列表
    storage_context=storage_context, # 使用之前创建的存储上下文
    embed_model=embedding, # 使用名为 embedding 的嵌入模型（假设之前已定义，代码中未给出）为节点生成嵌入向量
) # 创建一个 VectorStoreIndex 实例，用于基于向量数据库构建索引，以便进行高效的语义检索

# 构建检索器
from llama_index.core.retrievers import VectorIndexRetriever # 导入 LlamaIndex 库中的 VectorIndexRetriever 类，用于从向量索引中检索相关节点
# 想要自定义参数，可以构造参数字典
kwargs = {'similarity_top_k': 5, 'index': index, 'dimensions': 1024} # 创建一个字典，包含 VectorIndexRetriever 的自定义参数：检索最相似的 5 个节点，指定索引和向量维度
retriever = VectorIndexRetriever(**kwargs) # 使用自定义参数创建 VectorIndexRetriever 实例

# 构建合成器
from llama_index.core.response_synthesizers  import get_response_synthesizer # 导入 LlamaIndex 库中的 get_response_synthesizer 函数，用于创建响应合成器
response_synthesizer = get_response_synthesizer(llm=llm, streaming=True) # 创建一个响应合成器实例，使用之前定义的 llm 模型，并启用流式输出

# 构建问答引擎
from llama_index.core.query_engine import RetrieverQueryEngine # 导入 LlamaIndex 库中的 RetrieverQueryEngine 类，用于构建基于检索器的问答引擎
engine = RetrieverQueryEngine(
      retriever=retriever, # 使用之前创建的检索器
      response_synthesizer=response_synthesizer, # 使用之前创建的响应合成器
        ) # 创建一个 RetrieverQueryEngine 实例，将检索器和响应合成器组合在一起，形成一个可以回答问题的引擎
# 提问
question = "请问商标注册需要提供哪些文件？" # 定义一个问题
response = engine.query(question) # 使用问答引擎对问题进行查询
for text in response.response_gen: # 遍历响应的生成器，用于流式输出
    print(text, end="") # 打印流式生成的文本，不换行
# 配置查询工具
from llama_index.core.tools import QueryEngineTool # 导入 LlamaIndex 库中的 QueryEngineTool 类，用于将查询引擎封装成工具
from llama_index.core.tools import ToolMetadata # 导入 LlamaIndex 库中的 ToolMetadata 类，用于定义工具的元数据
query_engine_tools = [
    QueryEngineTool(
        query_engine=engine, # 使用之前创建的问答引擎
        metadata=ToolMetadata(
            name="RAG工具", # 定义工具的名称
            description=(
                "用于在原文中检索相关信息" # 定义工具的描述
            ),
        ),
    ),
] # 创建一个 QueryEngineTool 实例，将问答引擎封装成一个工具，并添加元数据

from llama_index.core.agent import ReActAgent # 导入 LlamaIndex 库中的 ReActAgent 类，用于构建 ReAct 代理
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True) # 创建一个 ReActAgent 实例，使用之前定义的查询引擎工具和 llm 模型，并启用详细输出
response1 = agent.chat("请问商标注册需要提供哪些文件？") # 使用 ReAct 代理对问题进行聊天
print(response1) # 打印代理的响应