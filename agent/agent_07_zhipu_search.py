import os  # 导入操作系统相关功能的模块
import json  # 导入 JSON 处理模块
import requests  # 导入 HTTP 请求库
from dotenv import load_dotenv  # 导入用于加载 .env 文件中环境变量的模块
from typing import List  # 导入类型提示模块中的 List
from zigent.agents import BaseAgent, ABCAgent  # 导入 zigent 库中的 BaseAgent 和 ABCAgent 类
from zigent.llm.agent_llms import LLM  # 导入 zigent 库中用于代理的 LLM 类
from zigent.commons import TaskPackage  # 导入 zigent 库中的 TaskPackage 类
from zigent.actions.BaseAction import BaseAction  # 导入 zigent 库中的 BaseAction 类
from zhipuai import ZhipuAI  # 导入智谱 AI 的 Python SDK
from datetime import datetime  # 导入 datetime 模块

# 加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量
# api_key = os.getenv('ZISHU_API_KEY1') 
api_key = os.getenv('ZISHU_API_KEY') # 从环境变量中获取名为 'ZISHU_API_KEY1' 的 API 密钥
base_url = "http://43.200.7.56:8008/v1"  # 注释掉的 base_url
chat_model = "Qwen2.5-72B"  # 注释掉的 chat_model
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')  # 从环境变量中获取智谱 AI 的 API 密钥
# base_url = "https://open.bigmodel.cn/api/paas/v4/"  # 定义 API 的基础 URL
# chat_model = "glm-4-flash"  # 定义使用的聊天模型名称

# 配置LLM
llm = LLM(api_key=api_key, base_url=base_url, model_name=chat_model)  # 创建 LLM 类的实例，用于与语言模型交互

# 定义Zhipu Web Search工具
def zhipu_web_search_tool(query: str) -> str:
    """
    使用智谱AI的GLM-4模型进行联网搜索，返回搜索结果的字符串。
    
    参数:
    - query: 搜索关键词

    返回:
    - 搜索结果的字符串形式
    """
    # 初始化客户端
    client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 使用智谱 AI 的 API 密钥初始化客户端

    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")  # 获取当前日期并格式化为 YYYY-MM-DD

    print("current_date:", current_date)  # 打印当前日期
    
    # 设置工具
    tools = [{
        "type": "web_search",
        "web_search": {
            "enable": True  # 启用联网搜索功能
        }
    }]  # 定义一个工具列表，包含一个 web_search 工具

    # 系统提示模板，包含时间信息
    system_prompt = f"""你是一个具备网络访问能力的智能助手，在适当情况下，优先使用网络信息（参考信息）来回答，
    以确保用户得到最新、准确的帮助。当前日期是 {current_date}。"""  # 定义系统提示，告知模型其具备联网能力和当前日期
        
    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},  # 系统角色消息，包含系统提示
        {"role": "user", "content": query}  # 用户角色消息，包含搜索查询
    ]
        
    # 调用API
    response = client.chat.completions.create(
        model="glm-4-flash",  # 指定使用的模型
        messages=messages,  # 传递消息列表
        tools=tools  # 传递工具列表
    )
    
    # 返回结果
    return response.choices[0].message.content  # 返回模型生成的回复内容

class ZhipuSearchAction(BaseAction):
    def __init__(self) -> None:
        action_name = "Zhipu_Search"  # 定义动作名称
        action_desc = "Using this action to search online content."  # 定义动作描述
        params_doc = {"query": "the search string. be simple."}  # 定义动作参数文档
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )  # 调用父类的初始化方法

    def __call__(self, query):
        results = zhipu_web_search_tool(query)  # 调用 zhipu_web_search_tool 函数执行搜索
        return results  # 返回搜索结果

class ZhipuSearchAgent(BaseAgent):
    def __init__(
        self,
        llm: LLM,  # 接收一个 LLM 实例
        actions: List[BaseAction] = [ZhipuSearchAction()],  # 接收一个动作列表，默认为包含 ZhipuSearchAction 的列表
        manager: ABCAgent = None,  # 接收一个 ABCAgent 实例，默认为 None
        **kwargs  # 接收额外的关键字参数
    ):
        name = "zhiu_search_agent"  # 定义代理名称
        role = "You can answer questions by using Zhipu search content."  # 定义代理角色
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            manager=manager,
        )  # 调用父类的初始化方法

def do_search_agent():
    # 创建代理实例
    search_agent = ZhipuSearchAgent(llm=llm)  # 创建 ZhipuSearchAgent 的实例，传入 LLM

    # 创建任务
    task = "2025年洛杉矶大火"  # 定义搜索任务
    task_pack = TaskPackage(instruction=task)  # 将任务封装到 TaskPackage 对象中

    # 执行任务并获取响应
    response = search_agent(task_pack)  # 调用代理执行任务
    print(response)  # 打印代理的响应

if __name__ == "__main__":
    do_search_agent()  # 当脚本作为主程序运行时，调用 do_search_agent 函数