import os
from dotenv import load_dotenv
import json
import re
import sqlite3
from llama_index.core.agent import ReActAgent  
from llama_index.core.tools import FunctionTool  
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings  
from llama_index.core.tools import QueryEngineTool   
from llama_index.core import SQLDatabase  
from llama_index.core.query_engine import NLSQLTableQueryEngine  
from sqlalchemy import create_engine, select
from openai import OpenAI
from pydantic import Field  # 导入Field，用于Pydantic模型中定义字段的元数据
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import List, Any, Generator  
# 创建数据库
sqllite_path = 'llmdb.db'  # 定义 SQLite 数据库文件的路径
con = sqlite3.connect(sqllite_path)  # 连接到指定的 SQLite 数据库，如果不存在则创建

# 创建表
sql = """
CREATE TABLE `section_stats` (
  `部门` varchar(100) DEFAULT NULL,
  `人数` int(11) DEFAULT NULL
);
"""  # 定义创建名为 `section_stats` 的表的 SQL 语句，包含 `部门` 和 `人数` 两个字段
c = con.cursor()  # 创建一个游标对象，用于执行 SQL 语句
# 检查表是否已存在，不存在则创建
cursor = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='section_stats';").fetchone()  # 执行 SQL 查询语句，检查名为 `section_stats` 的表是否存在
if cursor is None:  # 如果查询结果为空，表示表不存在
    c.execute(sql)  # 执行创建表的 SQL 语句
c.close()  # 关闭游标
con.close()  # 关闭数据库连接
con = sqlite3.connect(sqllite_path)  # 重新连接到 SQLite 数据库
c = con.cursor()  # 创建一个新的游标对象
data = [
    ["专利部",22],  # 包含部门名称和人数的列表
    ["商标部",25],  # 包含部门名称和人数的列表
]
for item in data:  # 遍历数据列表
    sql = """
    INSERT INTO section_stats (部门,人数)
    values('%s','%d')
    """%(item[0],item[1])  # 定义插入数据的 SQL 语句，使用字符串格式化将部门名称和人数插入
    c.execute(sql)  # 执行插入数据的 SQL 语句
    con.commit()  # 提交事务，将数据写入数据库
c.close()  # 关闭游标
con.close()  # 关闭数据库连接

# 加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY1')  # 从环境变量中获取名为 'ZISHU_API_KEY1' 的 API 密钥
if not api_key:  # 如果 API 密钥为空或未设置
    raise ValueError("环境变量 ZISHU_API_KEY1 未设置或为空，请检查 .env 文件！")  # 抛出异常，提示用户检查 .env 文件
base_url = "https://open.bigmodel.cn/api/paas/v4/"  # 定义 API 的基础 URL
chat_model = "glm-4-flash"  # 定义使用的聊天模型名称
emb_model = "embedding-3"  # 定义使用的嵌入模型名称 (虽然这里定义了，但后面代码中没有直接使用)

# 定义OurLLM类，继承自CustomLLM基类
class OurLLM(CustomLLM):  # 定义一个名为 OurLLM 的类，继承自 LlamaIndex 的 CustomLLM 类
    api_key: str = Field(default=api_key)  # 定义 api_key 字段，类型为字符串，默认值为从环境变量中获取的 api_key
    base_url: str = Field(default=base_url)  # 定义 base_url 字段，类型为字符串，默认值为定义的 base_url
    model_name: str = Field(default=chat_model)  # 定义 model_name 字段，类型为字符串，默认值为定义的 chat_model
    client: OpenAI = Field(default=None, exclude=True)  # 显式声明 client 字段，类型为 OpenAI 客户端对象，默认值为 None，并排除在序列化之外

    def __init__(self, api_key: str, base_url: str, model_name: str = chat_model, **data: Any):  # 定义类的初始化方法
        super().__init__(**data)  # 调用父类的初始化方法
        self.api_key = api_key  # 将传入的 api_key 赋值给类的 api_key 属性
        self.base_url = base_url  # 将传入的 base_url 赋值给类的 base_url 属性
        self.model_name = model_name  # 将传入的 model_name 赋值给类的 model_name 属性
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # 使用传入的 api_key 和 base_url 初始化 OpenAI 客户端实例

    @property
    def metadata(self) -> LLMMetadata:  # 定义获取 LLM 元数据的属性方法
        """Get LLM metadata."""  # 方法的文档字符串
        return LLMMetadata(  # 返回 LLMMetadata 对象
            model_name=self.model_name,  # 设置模型名称为类的 model_name 属性
        )

    @llm_completion_callback()  # 使用 llm_completion_callback 装饰器，用于跟踪 LLM 完成情况
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:  # 定义完成方法，接收提示词和额外的关键字参数
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}])  # 调用 OpenAI 客户端的 chat.completions.create 方法，发送提示词并获取响应
        if hasattr(response, 'choices') and len(response.choices) > 0:  # 检查响应是否包含 choices 属性且不为空
            response_text = response.choices[0].message.content  # 获取响应中的文本内容
            return CompletionResponse(text=response_text)  # 返回包含响应文本的 CompletionResponse 对象
        else:  # 如果响应格式不符合预期
            raise Exception(f"Unexpected response format: {response}")  # 抛出异常，提示响应格式错误

    @llm_completion_callback()  # 使用 llm_completion_callback 装饰器，用于跟踪 LLM 完成情况
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:  # 定义流式完成方法，接收提示词和额外的关键字参数，返回一个生成器
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True  # 设置 stream 参数为 True，启用流式传输
        )

        try:
            for chunk in response:  # 遍历响应中的每个数据块
                chunk_message = chunk.choices[0].delta  # 获取数据块中的消息增量
                if not chunk_message.content:  # 如果消息增量内容为空
                    continue  # 跳过当前循环
                content = chunk_message.content  # 获取消息增量的内容
                yield CompletionResponse(text=content, delta=content)  # 生成包含文本内容和增量的 CompletionResponse 对象

        except Exception as e:  # 捕获异常
            raise Exception(f"Unexpected response format: {e}")  # 抛出异常，提示响应格式错误

llm = OurLLM(api_key=api_key, base_url=base_url, model_name=chat_model)  # 创建 OurLLM 类的实例，传入 API 密钥、基础 URL 和模型名称
Settings.llm = llm  # 将创建的 LLM 实例设置为 LlamaIndex 的全局 LLM

## 创建数据库查询引擎
engine = create_engine("sqlite:///llmdb.db")  # 创建 SQLAlchemy 数据库引擎，连接到 SQLite 数据库

# prepare data
sql_database = SQLDatabase(engine, include_tables=["section_stats"])  # 创建 SQLDatabase 对象，指定要包含的表
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["section_stats"],
    llm=Settings.llm  # 将之前设置的全局 LLM 传递给查询引擎
)
# 创建工具函数
def multiply(a: float, b: float) -> float:
    """将两个数字相乘并返回乘积。"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)  # 从 multiply 函数创建 FunctionTool 对象

def add(a: float, b: float) -> float:
    """将两个数字相加并返回它们的和。"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)  # 从 add 函数创建 FunctionTool 对象

# 把数据库查询引擎封装到工具函数对象中
staff_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="section_staff",
    description="查询部门的人数。"
)  # 从查询引擎创建 QueryEngineTool 对象，并设置名称和描述

# 构建ReActAgent，可以加很多函数，在这里只加了加法函数和部门人数查询函数。
agent = ReActAgent.from_tools([add_tool, staff_tool], verbose=True)  # 创建 ReActAgent 实例，传入可用的工具列表，并启用详细输出

# 通过agent给出指令
response = agent.chat("请从数据库表中获取`专利部`和`商标部`的人数，并将这两个部门的人数相加！")  # 向 Agent 发送聊天指令
print(response)  # 打印 Agent 的响应