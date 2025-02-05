import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json
from zigent.llm.agent_llms import LLM
from zigent.actions import BaseAction, ThinkAct, FinishAct
from zigent.agents import BaseAgent
from zigent.commons import TaskPackage, AgentAct
from zigent.actions.InnerActions import INNER_ACT_KEY
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY1')
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-Air"

llm = LLM(api_key=api_key, base_url=base_url, model_name=chat_model)
class QuizGenerationAction(BaseAction):
    """Generate quiz questions from markdown content"""
    def __init__(self, llm: LLM) -> None:
        action_name = "GenerateQuiz"
        action_desc = "Generate quiz questions from markdown content"
        params_doc = {
            "content": "(Type: string): The markdown content to generate questions from",
            "question_types": "(Type: list): List of question types to generate",
            "audience": "(Type: string): Target audience for the quiz",
            "purpose": "(Type: string): Purpose of the quiz"
        }
        super().__init__(action_name, action_desc, params_doc)
        self.llm = llm
        
    def __call__(self, **kwargs):
        content = kwargs.get("content", "")
        question_types = kwargs.get("question_types", [])
        audience = kwargs.get("audience", "")
        purpose = kwargs.get("purpose", "")
        
        prompt = f"""
        你是一个辅助设计考卷的机器人,全程使用中文。
        你的任务是帮助用户快速创建、设计考卷，考卷以markdown格式给出。
        
        要求：
        1. 受众群体：{audience}
        2. 考察目的：{purpose}
        3. 需要包含以下题型：{", ".join(question_types)}
        4. 考卷格式要求：
        """
        prompt += """
        # 问卷标题
        ---
        1. 这是判断题的题干?
            - (x) True
            - ( ) False
        # (x)为正确答案

        2. 这是单选题的题干
            - (x) 这是正确选项
            - ( ) 这是错误选项
        # (x)为正确答案

        3. 这是多选题的题干?
            - [x] 这是正确选项1
            - [x] 这是正确选项2
            - [ ] 这是错误选项1
            - [ ] 这是错误选项2
        # [x]为正确答案

        4. 这是填空题的题干?
            - R:= 填空题答案
        #填空题正确答案格式
        """
        
        prompt += f"\n请根据以下内容生成考卷：\n{content}"
        
        quiz_content = self.llm.run(prompt)
        return {
            "quiz_content": quiz_content,
            "audience": audience,
            "purpose": purpose,
            "question_types": question_types
        }
class SaveQuizAction(BaseAction):
    """Save quiz to file and return URL"""
    def __init__(self) -> None:
        action_name = "SaveQuiz"
        action_desc = "Save quiz content to file and return URL"
        params_doc = {
            "quiz_content": "(Type: string): The quiz content to save",
            "quiz_title": "(Type: string): Title of the quiz"
        }
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        quiz_content = kwargs.get("quiz_content", "")
        quiz_title = kwargs.get("quiz_title", "quiz")
        
        output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{quiz_title}.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(quiz_content)
            
        return {
            "file_path": output_file,
            "quiz_url": f"/{output_file}"
        }

class QuizGeneratorAgent(BaseAgent):
    """Quiz generation agent that manages quiz creation process"""
    def __init__(
        self,
        llm: LLM,
        markdown_dir: str
    ):
        name = "QuizGeneratorAgent"
        role = """你是一个专业的考卷生成助手。你可以根据提供的Markdown内容生成结构良好、
        内容全面的考卷。你擅长根据受众群体和考察目的设计合适的题目。"""
        
        super().__init__(
            name=name,
            role=role,
            llm=llm,
        )
                
        self.markdown_dir = markdown_dir
        self.quiz_action = QuizGenerationAction(llm)
        self.save_action = SaveQuizAction()
        
        self._add_quiz_example()
        
    def _load_markdown_content(self) -> str:
        """Load all markdown files from directory"""
        content = []
        for root, _, files in os.walk(self.markdown_dir):
            for file in files:
                if file.endswith(".md"):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content.append(f.read())
        return "\n".join(content)
        
    def __call__(self, task: TaskPackage):
        """Process the quiz generation task"""
        # Parse task parameters
        params = json.loads(task.instruction)
        audience = params.get("audience", "")
        purpose = params.get("purpose", "")
        question_types = params.get("question_types", [])
        
        # Load markdown content
        content = self._load_markdown_content()
        
        # Generate quiz
        quiz_result = self.quiz_action(
            content=content,
            question_types=question_types,
            audience=audience,
            purpose=purpose
        )
        
        # Save quiz
        save_result = self.save_action(
            quiz_content=quiz_result["quiz_content"],
            quiz_title="generated_quiz"
        )
        
        task.answer = {
            "quiz_content": quiz_result["quiz_content"],
            "quiz_url": save_result["quiz_url"]
        }
        task.completion = "completed"
        
        return task
    def _add_quiz_example(self):
        """Add an illustration example for the quiz generator"""
        exp_task = json.dumps({
            "audience": "零基础",  # 水平
            "purpose": "测试Python基础知识掌握情况", # 目的
            "question_types": ["单选题", "多选题", "填空题"] # 题型
        })
        exp_task_pack = TaskPackage(instruction=exp_task)

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={INNER_ACT_KEY: """首先，我会加载Markdown内容，然后根据受众群体和考察目的生成考卷。"""}
        )
        obs_1 = "OK. 开始加载Markdown内容。"

        act_2 = AgentAct(
            name=self.quiz_action.action_name,
            params={
                "content": "Python基础内容...",
                "question_types": ["单选题", "多选题", "填空题"],
                "audience": "大学生",
                "purpose": "测试Python基础知识掌握情况"
            }
        )
        obs_2 = """# Python基础测试
        1. Python是什么类型的语言?
            - (x) 解释型
            - ( ) 编译型
        # (x)为正确答案"""

        act_3 = AgentAct(
            name=self.save_action.action_name,
            params={
                "quiz_content": obs_2,
                "quiz_title": "Python基础测试"
            }
        )
        obs_3 = {"file_path": "2025-01-15_03-37-40/Python基础测试.md", "quiz_url": "/2025-01-15_03-37-40/Python基础测试.md"}

        act_4 = AgentAct(
            name=FinishAct.action_name,
            params={INNER_ACT_KEY: "考卷生成并保存成功。"}
        )
        obs_4 = "考卷生成任务完成。"

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]

        self.prompt_gen.add_example(
            task=exp_task_pack,
            action_chain=exp_act_obs
        )
# 创建出题智能体
markdown_dir = "./wow-agent/docs"  # 指定包含Markdown文件的目录
agent = QuizGeneratorAgent(llm=llm, markdown_dir=markdown_dir)

# 定义考卷参数
quiz_params = {
    "audience": "零基础", # 受众群体
    "purpose": "测试基础知识掌握情况", # 考察目的
    "question_types": ["单选题"] # 需要包含的题型
}

# 生成考卷
task = TaskPackage(instruction=json.dumps(quiz_params))
result = agent(task)

print("生成的考卷内容：")
print(result.answer["quiz_content"])
print(f"考卷路径: {result.answer['quiz_url']}")

