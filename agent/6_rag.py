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
emb_model = "embedding-3"
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


from llama_index.core import SimpleDirectoryReader,Document
print(f"当前工作目录: {os.getcwd()}")
documents = SimpleDirectoryReader(input_files=['./wow-agent/docs/网红.txt']).load_data()

# 构建节点
from llama_index.core.node_parser import SentenceSplitter
transformations = [SentenceSplitter(chunk_size = 512)]

from llama_index.core.ingestion.pipeline import run_transformations
nodes = run_transformations(documents, transformations=transformations)

# 构建索引
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core import StorageContext, VectorStoreIndex

vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1024))
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes = nodes,
    storage_context=storage_context,
    embed_model=embedding,
)
# 构建检索器
from llama_index.core.retrievers import VectorIndexRetriever
# 想要自定义参数，可以构造参数字典
kwargs = {'similarity_top_k': 5, 'index': index} # 必要参数
retriever = VectorIndexRetriever(**kwargs)

# 构建合成器
from llama_index.core.response_synthesizers  import get_response_synthesizer
response_synthesizer = get_response_synthesizer(llm=llm, streaming=False)

# 构建问答引擎
from llama_index.core.query_engine import RetrieverQueryEngine
engine = RetrieverQueryEngine(
      retriever=retriever,
      response_synthesizer=response_synthesizer,
        )
'''
question = "如何延长网红周期？"
response = engine.query(question)
for text in response.response_gen:
    print(text, end="")
'''
    # 配置查询工具
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(
            name="RAG工具",
            description=(
                "用于在原文中检索相关信息"
            ),
        ),
    ),
]
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
response1 = agent.chat("如何延长网红周期")
print('res11:',response1)