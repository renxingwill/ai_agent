import os
from dotenv import load_dotenv
import json
import re
from typing import List
from zigent.agents import ABCAgent, BaseAgent
from zigent.llm.agent_llms import LLM
from zigent.commons import TaskPackage
from zigent.actions.BaseAction import BaseAction
from duckduckgo_search import DDGS

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
#api_key = os.getenv('ZISHU_API_KEY') # 从环境变量中获取名为 'ZISHU_API_KEY1' 的 API 密钥
#base_url = "http://43.200.7.56:8008/v1"  # 注释掉的 base_url
#chat_model = "Qwen2.5-72B"  # 注释掉的 chat_model
api_key = os.getenv('ZISHU_API_KEY2')
base_url = "http://49.0.255.97:3002/v1"
chat_model = "deepseek-reasoner"
#base_url = "https://open.bigmodel.cn/api/paas/v4/"
#chat_model = "glm-4-Air"
##base_url = "http://54.173.222.5:3582/v1/"
#chat_model = "gemini-2.0-flash-thinking-exp"
llm = LLM(api_key=api_key, base_url=base_url, model_name=chat_model)
from typing import List, Dict
from zigent.llm.agent_llms import LLM
from zigent.actions import BaseAction, ThinkAct, FinishAct
from zigent.agents import BaseAgent
from zigent.commons import TaskPackage, AgentAct
from zigent.actions.InnerActions import INNER_ACT_KEY
from datetime import datetime
import json
class WriteDirectoryAction(BaseAction):
    """Generate tutorial directory structure action"""
    def __init__(self) -> None:
        action_name = "WriteDirectory"
        action_desc = "Generate tutorial directory structure"
        params_doc = {
            "topic": "(Type: string): The tutorial topic name",
            "language": "(Type: string): Output language (default: 'Chinese')"
        }
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        topic = kwargs.get("topic", "")
        language = kwargs.get("language", "Chinese")
        
        directory_prompt = f"""
        请为主题"{topic}"生成教程目录结构,要求:
        1. 输出语言必须是{language}
        2. 严格按照以下字典格式输出: {{"title": "xxx", "directory": [{{"章节1": ["小节1", "小节2"]}}, {{"章节2": ["小节3", "小节4"]}}]}}
        3. 目录层次要合理,包含主目录和子目录
        4. 每个目录标题要有实际意义
        5. 不要有多余的空格或换行
        """
        
        # 调用 LLM 生成目录
        directory_data = llm.run({"prompt": directory_prompt})
        try:
            directory_data = json.loads(directory_data)
        except:
            directory_data = {"title": topic, "directory": []}
            
        return {
            "topic": topic,
            "language": language,
            "directory_data": directory_data
        }
class WriteContentAction(BaseAction):
    """Generate tutorial content action"""
    def __init__(self) -> None:
        action_name = "WriteContent"
        action_desc = "Generate detailed tutorial content based on directory structure"
        params_doc = {
            "title": "(Type: string): The section title",
            "chapter": "(Type: string): The chapter title",
            "directory_data": "(Type: dict): The complete directory structure", 
            "language": "(Type: string): Output language (default: 'Chinese')"
        }
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        title = kwargs.get("title", "")
        chapter = kwargs.get("chapter", "")
        language = kwargs.get("language", "Chinese")
        directory_data = kwargs.get("directory_data", {})
        
        content_prompt = f"""
        请为教程章节生成详细内容:
        教程标题: {directory_data.get('title', '')}
        章节: {chapter}
        小节: {title}
        
        要求:
        1. 内容要详细且准确
        2. 如果需要代码示例,请按标准规范提供
        3. 使用 Markdown 格式
        4. 输出语言必须是{language}
        5. 内容长度适中,通常在500-1000字之间
        """
        
        # 调用 LLM 生成内容
        content = llm.run({"prompt": content_prompt})
        return content
class TutorialAssistant(BaseAgent):
    """Tutorial generation assistant that manages directory and content creation"""
    def __init__(
        self,
        llm: llm,
        language: str = "Chinese"
    ):
        name = "TutorialAssistant"
        role = """You are a professional tutorial writer. You can create well-structured, 
        comprehensive tutorials on various topics. You excel at organizing content logically 
        and explaining complex concepts clearly."""
        
        super().__init__(
            name=name,
            role=role,
            llm=llm,
        )
        
        self.language = language
        self.directory_action = WriteDirectoryAction()
        self.content_action = WriteContentAction()
    
        # Add example for the tutorial assistant
        self._add_tutorial_example()
        
    def _generate_tutorial(self, directory_data: Dict) -> str:
        """Generate complete tutorial content based on directory structure"""
        full_content = []
        title = directory_data["title"]
        full_content.append(f"# {title}\n")
        
        # Generate table of contents
        full_content.append("## 目录\n")
        for idx, chapter in enumerate(directory_data["directory"], 1):
            for chapter_title, sections in chapter.items():
                full_content.append(f"{idx}. {chapter_title}")
                for section_idx, section in enumerate(sections, 1):
                    full_content.append(f"   {idx}.{section_idx}. {section}")
        full_content.append("\n---\n")
        
        # Generate content for each section
        for chapter in directory_data["directory"]:
            for chapter_title, sections in chapter.items():
                for section in sections:
                    content = self.content_action(
                        title=section,
                        chapter=chapter_title,
                        directory_data=directory_data,
                        language=self.language
                    )
                    full_content.append(content)
                    full_content.append("\n---\n")
        
        return "\n".join(full_content)

    def __call__(self, task: TaskPackage):
        """Process the tutorial generation task"""
        # Extract topic from task
        topic = task.instruction.split("Create a ")[-1].split(" tutorial")[0]
        if not topic:
            topic = task.instruction
            
        # Generate directory structure
        directory_result = self.directory_action(
            topic=topic,
            language=self.language
        )

        print(directory_result)
        
        # Generate complete tutorial
        tutorial_content = self._generate_tutorial(directory_result["directory_data"])

        # Save the result
        task.answer = tutorial_content
        task.completion = "completed"
        
        return task

    def _add_tutorial_example(self):
        """Add an illustration example for the tutorial assistant"""
        exp_task = "Create a Python tutorial for beginners"
        exp_task_pack = TaskPackage(instruction=exp_task)
        topic = "Python基础教程"

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={INNER_ACT_KEY: """First, I'll create a directory structure for the Python tutorial, 
            then generate detailed content for each section."""}
        )
        obs_1 = "OK. I'll start with the directory structure."

        act_2 = AgentAct(
            name=self.directory_action.action_name,
            params={
                "topic": topic, 
                "language": self.language
            }
        )
        obs_2 = """{"title": "Python基础教程", "directory": [
            {"第一章：Python介绍": ["1.1 什么是Python", "1.2 环境搭建"]},
            {"第二章：基础语法": ["2.1 变量和数据类型", "2.2 控制流"]}
        ]}"""

        act_3 = AgentAct(
            name=self.content_action.action_name,
            params={
                "title": "什么是Python",
                "chapter": "第一章：Python介绍",
                "directory_data": json.loads(obs_2),
                "language": self.language
            }
        )
        obs_3 = """# 第一章：Python介绍\n## 什么是Python\n\nPython是一种高级编程语言..."""

        act_4 = AgentAct(
            name=FinishAct.action_name,
            params={INNER_ACT_KEY: "Tutorial structure and content generated successfully."}
        )
        obs_4 = "Tutorial generation task completed successfully."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
        
        self.prompt_gen.add_example(
            task=exp_task_pack,
            action_chain=exp_act_obs
        )
if __name__ == "__main__":
    assistant = TutorialAssistant(llm=llm)

     # 交互式生成教程
    FLAG_CONTINUE = True
    while FLAG_CONTINUE:
        input_text = input("What tutorial would you like to create?\n")
        task = TaskPackage(instruction=input_text)
        result = assistant(task)
        print("\nGenerated Tutorial:\n")
        print(result.answer)

        # 创建输出目录
        output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存文件
        output_file = os.path.join(output_dir, f"{input_text}.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.answer)
        if input("\nDo you want to create another tutorial? (y/n): ").lower() != "y":
            FLAG_CONTINUE = False

