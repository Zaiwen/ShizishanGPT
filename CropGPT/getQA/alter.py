# -*- coding: utf-8 -*-

import pandas as pd
import json

def export_modified_conversations_to_json(df, num_records, file_name, col_list):
    """
    将对话数据以修改后的格式导出到 JSON 文件。

    :param df: 包含对话数据的 DataFrame。
    :param num_records: 要导出的记录数。
    :param file_name: 输出 JSON 文件的名称。
    :param col_list: 要导出的列
    """
    output = []

    # 遍历 DataFrame 并构建修改后所需的数据结构
    for i, row in df.iterrows():
        # 将所有字段转为 JSON 可序列化的类型
        row_data = {}
        for col in col_list:
            value = row[col]
            if isinstance(value, pd.np.int64):  # 如果是 int64 类型
                value = int(value)  # 转换为 int 类型
            row_data[col] = value

        output.append({
            "instruction": "用户指令（必填）",
            "input": row_data['input'],
            "output": row_data['output']
        })

    # 将列表转换为 JSON 格式并保存为文件
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

# 读取数据并确保没有丢失列名
data = pd.read_csv("questions_answers.csv", encoding='utf-8', header=None)
data.columns = ['input', 'output']  # 设定列名
print(data)

# 调用函数导出 JSON
export_modified_conversations_to_json(data, data.shape[0], './AgriHubHZAU.json', data.columns[0:2])
