import json
import requests
BASE_URL = "http://49.0.258.97:3002/v1/chat/completions"

payload = {    
    "messages": [
        {
            "role": "user",
            "content": "请写一篇1000字左右的文章，论述法学专业的就业前景。"
        }
    ],
    "model": "deepseek-reasoner",
    
}

response = requests.post(BASE_URL, json=payload, stream=True) # stream=True for requests
print("res:",response.text)

full_response_text = "" # 用于拼接完整回复文本

if response.status_code == 200:
    for line in response.iter_lines(): # 逐行迭代响应内容
        if line: # 检查是否有内容 (去除空行)
            try:
                json_data = json.loads(line) # 解析 JSON 对象
                if 'message' in json_data and 'content' in json_data['message']:
                    content_chunk = json_data['message']['content']
                    print(content_chunk, end="", flush=True) # 逐字打印，不换行，实时显示
                    full_response_text += content_chunk # 拼接文本

                if json_data.get('done', False): # 检查是否完成
                    break # 完成后跳出循环

            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(response.text)

print("\n\n完整回复文本:\n" + full_response_text) # 打印完整回复文本