import os
from dotenv import load_dotenv
import json
import re

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
# 从环境变量中读取api_key
import os
api_key = os.getenv('ZISHU_API_KEY1')
if not api_key:
    raise ValueError("环境变量 ZISHU_API_KEY1 未设置或为空，请检查 .env 文件！")
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"
emb_model = "embedding-3"
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
# 定义OurLLM类，继承自CustomLLM基类
class OurLLM(CustomLLM):
    # 定义api_key字段，类型为字符串，默认值为从环境变量中读取的api_key
    api_key: str = Field(default=api_key)
    # 定义base_url字段，类型为字符串，默认值为指定的API基础URL
    base_url: str = Field(default=base_url)
    # 定义model_name字段，类型为字符串，默认值为指定的聊天模型名称
    model_name: str = Field(default=chat_model)
    # 定义client字段，类型为OpenAI客户端对象，默认值为None，并且在序列化时排除此字段
    client: OpenAI = Field(default=None, exclude=True)  # 显式声明 client 字段

    # 初始化方法，用于创建OurLLM类的实例
    def __init__(self, api_key: str, base_url: str, model_name: str = chat_model, **data: Any):
        # 调用父类CustomLLM的初始化方法
        super().__init__(**data)
        # 将传入的api_key赋值给实例的api_key属性
        self.api_key = api_key
        # 将传入的base_url赋值给实例的base_url属性
        self.base_url = base_url
        # 将传入的model_name赋值给实例的model_name属性
        self.model_name = model_name
        # 使用传入的api_key和base_url初始化 OpenAI 客户端实例，用于与API进行交互
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)  # 使用传入的api_key和base_url初始化 client 实例

    # 定义一个属性方法，用于获取LLM的元数据
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 返回一个LLMMetadata对象，其中包含模型名称
        return LLMMetadata(
            model_name=self.model_name,
        )

    # 定义一个方法，用于完成文本生成，并应用回调函数
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 使用OpenAI客户端调用chat.completions.create方法，传入模型名称和用户消息
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        # 检查响应是否包含choices属性且长度大于0
        if hasattr(response, 'choices') and len(response.choices) > 0:
            # 从响应中提取生成的文本内容
            response_text = response.choices[0].message.content
            # 返回一个CompletionResponse对象，包含生成的文本
            return CompletionResponse(text=response_text)
        else:
            # 如果响应格式不符合预期，则抛出异常
            raise Exception(f"Unexpected response format: {response}")

    # 定义一个方法，用于流式完成文本生成，并应用回调函数
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        # 使用OpenAI客户端调用chat.completions.create方法，传入模型名称、用户消息和stream=True以启用流式传输
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        try:
            # 遍历响应中的每个数据块
            for chunk in response:
                # 获取当前数据块中的消息增量
                chunk_message = chunk.choices[0].delta
                # 如果消息增量中没有内容，则跳过
                if not chunk_message.content:
                    continue
                # 获取消息增量的内容
                content = chunk_message.content
                # 生成一个CompletionResponse对象，包含文本内容和增量内容，并通过yield返回
                yield CompletionResponse(text=content, delta=content)

        except Exception as e:
            # 如果在处理流式响应时发生异常，则抛出异常
            raise Exception(f"Unexpected response format: {e}")

# 创建OurLLM类的实例，传入API密钥、基础URL和模型名称
llm = OurLLM(api_key=api_key, base_url=base_url, model_name=chat_model)
# 调用llm实例的stream_complete方法，传入提示语“你是谁？”，并将返回的生成器对象赋值给response变量
response = llm.stream_complete("你是谁？")
#print('res:',response)
for chunk in response:
    print(chunk, end="", flush=True)
import sys
import os

# 将当前脚本所在目录的父目录的父目录添加到Python的模块搜索路径中，以便导入上层目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# 定义一个函数，用于将两个数字相乘并返回结果
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

# 定义一个函数，用于将两个数字相加并返回结果
def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

# 定义主函数
def main():

    # 使用FunctionTool.from_defaults方法创建一个工具，该工具封装了multiply函数
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    # 使用FunctionTool.from_defaults方法创建一个工具，该工具封装了add函数
    add_tool = FunctionTool.from_defaults(fn=add)

    # 创建ReActAgent实例，传入工具列表、LLM模型和verbose=True以启用详细输出
    agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

    # 使用agent的chat方法与模型进行对话，传入问题并要求使用工具计算每一步
    response = agent.chat("20+（2*4）等于多少？使用工具计算每一步")

    # 打印agent的响应
    print(response)

# 当脚本作为主程序运行时，执行main函数
if __name__ == "__main__":
    main()