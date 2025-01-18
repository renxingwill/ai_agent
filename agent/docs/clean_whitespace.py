import os

def remove_whitespace_from_file(filepath):
    """
    移除指定文件中的空格和换行符。

    Args:
        filepath: 要处理的文件的完整路径。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 移除空格和换行符
        cleaned_content = content.replace(" ", "").replace("\n", "")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"已成功处理文件: {filepath}")

    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

if __name__ == "__main__":
    file_path_input = input("请输入 '营销.txt' 文件的完整路径 (例如: D:\\test2\\agent\\wow-agent\\docs\\营销.txt): ")

    # 自动添加转义字符 (如果需要)
    # Windows 路径可以直接使用，不需要额外添加转义，
    # 但为了兼容性，可以保留或移除这部分。
    # escaped_file_path = os.path.normpath(file_path_input) 

    filepath = file_path_input # 直接使用用户输入的路径

    if os.path.exists(filepath):
        remove_whitespace_from_file(filepath)
    else:
        print(f"错误: 文件 '{filepath}' 不存在。")