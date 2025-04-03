import os
from dotenv import load_dotenv
import json
from typing import List, Dict
from zigent.agents import BaseAgent
from zigent.llm.agent_llms import LLM_NEWAPI
from zigent.commons import TaskPackage, AgentAct
from zigent.actions import BaseAction, ThinkAct, FinishAct
from zigent.actions.InnerActions import INNER_ACT_KEY
from datetime import datetime
import time
# Load environment variables
load_dotenv()
api_key = os.getenv('ONE_API_KEYK')
base_url = "https://wilful-cicily-zhongdadianzi-8b661d5f.koyeb.app/v1/"
chat_model = "gemini-2.5-pro-exp-03-25"
llm = LLM_NEWAPI(api_key=api_key, base_url=base_url, model_name=chat_model)

class PlanStoryAction(BaseAction):
    """Generate story planning action"""
    def __init__(self) -> None:
        action_name = "PlanStory"
        action_desc = "Generate overall story planning including protagonist, structure, and plot development"
        params_doc = {
            "topic": "(Type: string): The novel topic or theme",
            "language": "(Type: string): Output language (default: 'Chinese')"
        }
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        topic = kwargs.get("topic", "")
        language = kwargs.get("language", "Chinese")
        
        planning_prompt = f"""
        请为主题"{topic}"生成小说的整体故事规划，要求：
        1. 输出语言必须是{language}
        2. 包含主角介绍、故事结构和情节发展规划
        3. 严格按照以下字典格式输出：{{"protagonist": "xxx", "structure": "xxx", "plot": "xxx"}}
        4. 确保内容有创意且吸引人
        """
        
        planning_data = llm.run({"prompt": planning_prompt})
        try:
            planning_data = json.loads(planning_data[7:-4])
        except:
            planning_data = {"protagonist": "", "structure": "", "plot": ""}
        return planning_data

class WriteNovelDirectoryAction(BaseAction):
    """Generate novel directory structure action"""
    def __init__(self) -> None:
        action_name = "WriteNovelDirectory"
        action_desc = "Generate novel chapter directory structure"
        params_doc = {
            "topic": "(Type: string): The novel topic",
            "planning": "(Type: dict): Story planning data",
            "language": "(Type: string): Output language (default: 'Chinese')"
        }
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        topic = kwargs.get("topic", "")
        planning = kwargs.get("planning", {})
        language = kwargs.get("language", "Chinese")
        
        directory_prompt = f"""
        请为主题"{topic}"生成小说章节目录结构，要求：
        1. 输出语言必须是{language}
        2. 基于故事规划：{planning}
        3. 严格按照以下字典格式输出：{{"title": "xxx", "chapters": ["章节1", "章节2", ...]}}
        4. 章节标题要有吸引力
        5. 目录章节不超过4个。
        """
        
        directory_data = llm.run({"prompt": directory_prompt})
        try:
            directory_data = json.loads(directory_data[7:-4])
        except:
            directory_data = {"title": topic, "chapters": []}
        return directory_data

class WriteChapterAction(BaseAction):
    """Generate novel chapter content action"""
    def __init__(self) -> None:
        action_name = "WriteChapter"
        action_desc = "Generate detailed novel chapter content"
        params_doc = {
            "chapter": "(Type: string): The chapter title",
            "prev_summary": "(Type: string): Summary of the previous chapter",
            "next_summary": "(Type: string): Summary of the next chapter",
            "directory_data": "(Type: dict): The complete directory structure",
            "language": "(Type: string): Output language (default: 'Chinese')"
        }
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        chapter = kwargs.get("chapter", "")
        prev_summary = kwargs.get("prev_summary", "")
        next_summary = kwargs.get("next_summary", "")
        directory_data = kwargs.get("directory_data", {})
        language = kwargs.get("language", "Chinese")
        
        content_prompt = f"""
        请为小说章节生成详细内容：
        小说标题：{directory_data.get('title', '')}
        章节：{chapter}
        上一章节摘要：{prev_summary}
        下一章节摘要：{next_summary}
        
        要求：
        1. 内容要引人入胜，情节紧凑
        2. 确保与前后章节连贯
        3. 使用文学化的语言
        4. 输出语言必须是{language}
        5. 内容长度适中，通常在1000-2000字之间
        """
        
        content = llm.run({"prompt": content_prompt})
        return content

class NovelAssistant(BaseAgent):
    """Novel generation assistant that manages story planning, directory, and content creation"""
    def __init__(
        self,
        llm: LLM_NEWAPI,
        language: str = "Chinese"
    ):
        name = "NovelAssistant"
        role = """You are a professional novelist. You can create engaging, 
        well-structured novels on various topics. You excel at developing 
        compelling characters and plots."""
        
        super().__init__(
            name=name,
            role=role,
            llm=llm,
        )
        
        self.language = language
        self.plan_action = PlanStoryAction()
        self.directory_action = WriteNovelDirectoryAction()
        self.content_action = WriteChapterAction()
    
        # Add example for the novel assistant
        self._add_novel_example()
        
    def _generate_novel(self, directory_data: Dict, planning: Dict) -> str:
        """Generate complete novel content based on directory structure"""
        full_content = []
        title = directory_data["title"]
        full_content.append(f"# {title} #\n")
        
        # Generate content for each chapter
        chapters = directory_data["chapters"]
        for i, chapter in enumerate(chapters):
            prev_summary = self._get_summary(chapters, i-1) if i > 0 else "无"
            next_summary = self._get_summary(chapters, i+1) if i < len(chapters)-1 else "无"
            content = self.content_action(
                chapter=chapter,
                prev_summary=prev_summary,
                next_summary=next_summary,
                directory_data=directory_data,
                language=self.language
            )
            full_content.append(f"## {chapter} ##\n")
            full_content.append(content)
            full_content.append("\n---\n")
            time.sleep(14)
        
        return "\n".join(full_content)
    
    def _get_summary(self, chapters: List[str], index: int) -> str:
        """Generate summary for a chapter"""
        if 0 <= index < len(chapters):
            summary_prompt = f"请为章节'{chapters[index]}'生成简短摘要。"
            summary = llm.run({"prompt": summary_prompt})
            time.sleep(15)
            return summary
        return "无"

    def __call__(self, task: TaskPackage):
        """Process the novel generation task"""
        topic = task.instruction.split("Create a ")[-1].split(" novel")[0]
        if not topic:
            topic = task.instruction
            
        # Generate story planning
        planning = self.plan_action(topic=topic, language=self.language)
        
        # Generate directory structure
        directory_result = self.directory_action(
            topic=topic,
            planning=planning,
            language=self.language
        )
        
        # Generate complete novel
        novel_content = self._generate_novel(directory_result, planning)

        # Save the result
        task.answer = novel_content
        task.completion = "completed"
        
        return task

    def _add_novel_example(self):
        """Add an illustration example for the novel assistant"""
        exp_task = "Create a mystery novel set in Victorian London"
        exp_task_pack = TaskPackage(instruction=exp_task)
        topic = "Victorian London Mystery"

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={INNER_ACT_KEY: """First, I'll plan the story, then create the chapter directory, 
            and finally generate the content for each chapter."""}
        )
        obs_1 = "OK. I'll start with the story planning."

        act_2 = AgentAct(
            name=self.plan_action.action_name,
            params={"topic": topic, "language": self.language}
        )
        obs_2 = """{"protagonist": "Detective John Smith", "structure": "Three-act structure", "plot": "A series of mysterious murders..."}"""

        act_3 = AgentAct(
            name=self.directory_action.action_name,
            params={"topic": topic, "planning": json.loads(obs_2), "language": self.language}
        )
        obs_3 = """{"title": "Shadows of London", "chapters": ["Chapter 1: The First Victim", "Chapter 2: Clues in the Fog", "Chapter 3: The Final Confrontation"]}"""

        act_4 = AgentAct(
            name=self.content_action.action_name,
            params={
                "chapter": "Chapter 1: The First Victim",
                "prev_summary": "无",
                "next_summary": "Detective Smith finds a crucial clue...",
                "directory_data": json.loads(obs_3),
                "language": self.language
            }
        )
        obs_4 = """## Chapter 1: The First Victim ##\n\nThe fog was thick that night..."""

        act_5 = AgentAct(
            name=FinishAct.action_name,
            params={INNER_ACT_KEY: "Novel structure and content generated successfully."}
        )
        obs_5 = "Novel generation task completed successfully."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4), (act_5, obs_5)]
        
        self.prompt_gen.add_example(
            task=exp_task_pack,
            action_chain=exp_act_obs
        )

if __name__ == "__main__":
    assistant = NovelAssistant(llm=llm)

    # Interactive novel generation
    FLAG_CONTINUE = True
    while FLAG_CONTINUE:
        input_text = input("What kind of novel would you like to create?\n")
        task = TaskPackage(instruction=input_text)
        result = assistant(task)
        print("\nGenerated Novel:\n")
        print(result.answer)

        # Create output directory
        output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save file
        output_file = os.path.join(output_dir, f"{input_text}.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.answer)
        if input("\nDo you want to create another novel? (y/n): ").lower() != "y":
            FLAG_CONTINUE = False