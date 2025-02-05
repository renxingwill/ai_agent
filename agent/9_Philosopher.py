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
api_key = os.getenv('ZISHU_API_KEY') # 从环境变量中获取名为 'ZISHU_API_KEY1' 的 API 密钥
base_url = "http://43.200.7.56:8008/v1"  # 注释掉的 base_url
chat_model = "Qwen2.5-72B"  # 注释掉的 chat_model
#api_key = os.getenv('ZISHU_API_KEY1')
#base_url = "https://open.bigmodel.cn/api/paas/v4/"
#chat_model = "glm-4-flash"
llm = LLM(api_key=api_key, base_url=base_url, model_name=chat_model)
from typing import List
from zigent.actions.BaseAction import BaseAction
from zigent.agents import ABCAgent, BaseAgent

# 定义 Philosopher 类，继承自 BaseAgent 类
class Philosopher(BaseAgent):
    def __init__(
        self,
        philosopher,
        llm: llm,
        actions: List[BaseAction] = [], 
        manager: ABCAgent = None,
        **kwargs
    ):
        name = philosopher
        # 角色
        role = f"""You are {philosopher}, the famous educator in history. You are very familiar with {philosopher}'s Book and Thought. Tell your opinion on behalf of {philosopher}."""
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            manager=manager
        )

# 初始化哲学家对象
Confucius = Philosopher(philosopher= "Confucius", llm = llm) # 孔子
Socrates = Philosopher(philosopher="Socrates", llm = llm) # 苏格拉底
Aristotle = Philosopher(philosopher="Aristotle", llm = llm) # 亚里士多德
from zigent.commons import AgentAct, TaskPackage
from zigent.actions import ThinkAct, FinishAct
from zigent.actions.InnerActions import INNER_ACT_KEY
from zigent.agents.agent_utils import AGENT_CALL_ARG_KEY

# 为哲学家智能体添加示例任务
# 设置示例任务:询问生命的意义
exp_task = "What do you think the meaning of life?"
exp_task_pack = TaskPackage(instruction=exp_task)

# 第一个动作:思考生命的意义
act_1 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""Based on my thought, we are born to live a meaningful life, and it is in living a meaningful life that our existence gains value. Even if a life is brief, if it holds value, it is meaningful. A life without value is merely existence, a mere survival, a walking corpse."""
    },
)
# 第一个动作的观察结果
obs_1 = "OK. I have finished my thought, I can pass it to the manager now."

# 第二个动作:总结思考结果
act_2 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "I can summarize my thought now."})
# 第二个动作的观察结果
obs_2 = "I finished my task, I think the meaning of life is to pursue value for the whold world."
# 将动作和观察组合成序列
exp_act_obs = [(act_1, obs_1), (act_2, obs_2)]

# 为每个哲学家智能体添加示例
# 为孔子添加示例
Confucius.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)
# 为苏格拉底添加示例
Socrates.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)
# 为亚里士多德添加示例
Aristotle.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)
# 定义管理者代理
from zigent.agents import ManagerAgent

# 设置管理者代理的基本信息
manager_agent_info = {
    "name": "manager_agent",
    "role": "you are managing Confucius, Socrates and Aristotle to discuss on questions. Ask their opinion one by one and summarize their view of point."
}
# 设置团队成员
team = [Confucius, Socrates, Aristotle]
# 创建管理者代理实例
manager_agent = ManagerAgent(name=manager_agent_info["name"], role=manager_agent_info["role"], llm=llm, TeamAgents=team)

# 为管理者代理添加示例任务
exp_task = "What is the meaning of life?"
exp_task_pack = TaskPackage(instruction=exp_task)

# 第一步：思考如何处理任务
act_1 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""I can ask Confucius, Socrates and Aristotle one by one on their thoughts, and then summary the opinion myself."""
    },
)
obs_1 = "OK."

# 第二步：询问孔子的观点
act_2 = AgentAct(
    name=Confucius.name,
    params={AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
        },
)
obs_2 = """Based on my thought, I think the meaning of life is to pursue value for the whold world."""

# 第三步：思考下一步行动
act_3 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""I have obtained information from Confucius, I need to collect more information from Socrates."""
    },
)
obs_3 = "OK."

# 第四步：询问苏格拉底的观点
act_4 = AgentAct(
    name=Socrates.name,
    params={AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
        },
)
obs_4 = """I think the meaning of life is finding happiness."""

# 第五步：继续思考下一步
act_5 = AgentAct(
    name=ThinkAct.action_name,
    params={INNER_ACT_KEY: f"""I have obtained information from Confucius and Socrates, I can collect more information from Aristotle."""
    },
)
obs_5 = "OK."

# 第六步：询问亚里士多德的观点
act_6 = AgentAct(
    name=Aristotle.name,
    params={AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
        },
)
obs_6 = """I believe the freedom of spirit is the meaning."""

# 最后一步：总结所有观点
act_7 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Their thought on the meaning of life is to pursue value, happiniss and freedom of spirit."})
obs_7 = "Task Completed. The meaning of life is to pursue value, happiness and freedom of spirit."

# 将所有动作和观察组合成序列
exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4), (act_5, obs_5), (act_6, obs_6), (act_7, obs_7)]

# 将示例添加到管理者代理的提示生成器中
manager_agent.prompt_gen.add_example(
    task = exp_task_pack, action_chain = exp_act_obs
)
from zigent.commons import AgentAct, TaskPackage

exp_task = "先有鸡还是先有蛋?"
exp_task_pack = TaskPackage(instruction=exp_task)
manager_agent(exp_task_pack)

