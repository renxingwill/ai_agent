import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key


api_key = os.getenv('ZISHU_API_KEY1')
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"

from openai import OpenAI
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)
def get_completion(prompt):
    response = client.chat.completions.create(        
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=chat_model,  # 填写需要调用的模型名称
    )
    print("res:",response)
    return response.choices[0].message.content
response = get_completion("人工智能未来5年会替代哪些职业？")
print("result:",response)