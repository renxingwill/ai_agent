import os
from dotenv import load_dotenv
import json
import re
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import List
from zigent.agents import ABCAgent, BaseAgent
from zigent.llm.agent_llms import LLM
from zigent.commons import TaskPackage
from zigent.actions.BaseAction import BaseAction
from duckduckgo_search import DDGS

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY1')
if not api_key:
    raise ValueError("环境变量 ZISHU_API_KEY1 未设置或为空，请检查 .env 文件！")    
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"
emb_model = "embedding-2"
llm = LLM(api_key=api_key, base_url=base_url, model_name=chat_model)
# response = llm.run("你是谁？")
# print(response)
class DuckSearchAction(BaseAction):
    def __init__(self) -> None:
        action_name = "DuckDuckGo_Search"
        action_desc = "Using this action to search online content."
        params_doc = {"query": "the search string. be simple."}
        self.ddgs = DDGS()
        super().__init__(
            action_name=action_name, 
            action_desc=action_desc, 
            params_doc=params_doc,
        )

    def __call__(self, query):
        results = self.ddgs.text(query)
        return results
search_action = DuckSearchAction()
#results = search_action("2025年西藏地震")
#print(results)
class DuckSearchAgent(BaseAgent):
    def __init__(
        self,
        llm: llm,
        actions: List[BaseAction] = [DuckSearchAction()],
        manager: ABCAgent = None,
        **kwargs
    ):
        name = "duck_search_agent"
        role = "You can answer questions by using duck duck go search content."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            manager=manager,
        )
def do_search_agent():
    # 创建代理实例
    search_agent = DuckSearchAgent(llm=llm)

    # 创建任务
    task = "2025年西藏地震"
    task_pack = TaskPackage(instruction=task)

    # 执行任务并获取响应
    response = search_agent(task_pack)
    print("response:", response)

if __name__ == "__main__":
    do_search_agent()

