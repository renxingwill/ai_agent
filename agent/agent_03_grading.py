import os
from dotenv import load_dotenv
import json
import re

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY1')
if not api_key:
    raise ValueError("环境变量 ZISHU_API_KEY1 未设置或为空，请检查 .env 文件！")    
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "GLM-4-Air"
from openai import OpenAI
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)

def extract_json_content(text):
    # 这个函数的目标是提取大模型输出内容中的json部分，并对json中的换行符、首位空白符进行删除
    text = text.replace("\n","")
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text

class JsonOutputParser:
    def parse(self, result):
        # 这个函数的目标是把json字符串解析成python对象
        # 其实这里写的这个函数性能很差，经常解析失败，有很大的优化空间
        try:
            result = extract_json_content(result)
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid json output: {result}") from e

class GradingOpenAI:
    def __init__(self):
        self.model = "glm-4-flash"
        self.output_parser = JsonOutputParser()
        self.template = """你是一位中国专利代理师考试阅卷专家，
擅长根据给定的题目和答案为考生生成符合要求的评分和中文评语，
并按照特定的格式输出。
你的任务是，根据我输入的考题和答案，针对考生的作答生成评分和中文的评语，并以JSON格式返回。
阅卷标准适当宽松一些，只要考生回答出基本的意思就应当给分。
答案如果有数字标注，含义是考生如果答出这个知识点，这道题就会得到几分。
生成的中文评语需要能够被json.loads()这个函数正确解析。
生成的整个中文评语需要用英文的双引号包裹，在被包裹的字符串内部，请用中文的双引号。
中文评语中不可以出现换行符、转义字符等等。

输出格式为JSON:
{{
  "llmgetscore": 0,
  "llmcomments": "中文评语"
}}

比较学生的回答与正确答案，
并给出满分为10分的评分和中文评语。 
题目：{ques_title} 
答案：{answer} 
学生的回复：{reply}"""

    def create_prompt(self, ques_title, answer, reply):
        return self.template.format(
            ques_title=ques_title,
            answer=answer,
            reply=reply
        )

    def grade_answer(self, ques_title, answer, reply):
        success = False
        while not success:
            # 这里是一个不得已的权宜之计
            # 上面的json解析函数不是表现很差吗，那就多生成几遍，直到解析成功
            # 对大模型生成的内容先解析一下，如果解析失败，就再让大模型生成一遍
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位专业的考试阅卷专家。"},
                        {"role": "user", "content": self.create_prompt(ques_title, answer, reply)}
                    ],
                    temperature=0.7
                )
                print('response:',response)

                result = self.output_parser.parse(response.choices[0].message.content)
                success = True
            except Exception as e:
                print(f"Error occurred: {e}")
                continue

        return result['llmgetscore'], result['llmcomments']

    def run(self, input_data):
        output = []
        for item in input_data:
            score, comment = self.grade_answer(
                item['ques_title'], 
                item['answer'], 
                item['reply']
            )
            item['llmgetscore'] = score
            item['llmcomments'] = comment
            output.append(item)
        return output
grading_openai = GradingOpenAI()
import json
import os

input_data = []
file_path = './docs/content.json'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                input_data = data
            else:
                print(f"Warning: '{file_path}' does not contain a list at the top level. Attempting to process as individual JSON objects.")
                # Attempt to split and load individual JSON objects
                lines = content.strip().split('\n')
                temp_list = []
                for line in lines:
                    try:
                        obj = json.loads(line)
                        temp_list.append(obj)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON object: {line}. Error: {e}")
                if temp_list:
                    input_data = temp_list
                else:
                    print(f"Error: Could not parse any valid JSON objects from '{file_path}'.")

        except json.JSONDecodeError as e:
            print(f"Error: '{file_path}' is not a valid JSON file. Attempting to process as individual JSON objects per line.")
            # Attempt to process as individual JSON objects per line if the whole file is not valid JSON
            lines = content.strip().split('\n')
            temp_list = []
            for line in lines:
                try:
                    obj = json.loads(line)
                    temp_list.append(obj)
                except json.JSONDecodeError as line_e:
                    print(f"Error decoding JSON object: {line}. Error: {line_e}")
            if temp_list:
                input_data = temp_list
            else:
                print(f"Error: Could not parse any valid JSON objects from '{file_path}'.")

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found in the current directory.")
print('inputdata:',input_data)

graded_data = grading_openai.run(input_data)
print('评判结果：',graded_data)