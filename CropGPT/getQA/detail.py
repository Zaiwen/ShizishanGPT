import csv
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_llm_model
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2700,  # 块的大小
    chunk_overlap=100  # 重叠部分
)

# 获取 LLM 模型
llm = get_llm_model()

# 提示模板（保留原始提示）
prompt_str = """
#01 你是数据处理专家。
#02 你需要根据用户信息，提取5个左右的问答对，记住问题和答案都来源于用户信息，请尽量不要自行编造问答对！
#03 问题要宽泛一点，不要针对某一具体文献来提问。避免提及具体的书名、作者、出版信息等。例如，“园林绿化苗木繁育时需要注意哪些方面？”是合适的，而“这本书的出版信息中提到了哪些内容”则不合适。
#04 答案要全面，多使用我的信息，内容要更丰富，并尽量涵盖广泛的应用场景和实际情况。
#05 你必须根据我的对示例格式来生成：
#06 示例格式:
用户消息:"据了解干旱能够消耗玉米的水分、光照太强能够损失玉米的组织，从而影响玉米生长。"s
输出:
1、干旱如何影响玉米的生长？
答:干旱能够消耗玉米的水分。
2、光照对玉米生长的影响是什么？
答:光照太强会损害玉米组织，光照需要事宜。


#07 真实用户消息如下\n
用户信息:{User_Msg}
输出:
"""
# 使用正则表达式提取问答对
qa_pattern = re.compile(r'(\d+、[^？]+？)\s*答:(.*?)\s*(?=\d+、|$)', re.S)
# 处理单个文本块的函数
def process_chunk(chunk, chain):
    try:
        input_data = {"User_Msg": chunk}
        response = chain.invoke(input_data)['text']
        return response
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None

# 处理所有文本块
def bornQA_repair(chunks):
    prompt = PromptTemplate.from_template(prompt_str)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    # qa_pairs = []
    for chunk in chunks:
        # print(chunk)
        response = process_chunk(chunk, chain)
        if response:
            qa_pairs = re.findall(qa_pattern, response)
            qa_pairs = [(re.sub(r'^\d+、', '', question).strip(), answer.strip()) for question, answer in qa_pairs]
            # 将数据写入 CSV 文件
            with open('questions_answers.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for question, answer in qa_pairs:
                    if len(question)>=1 and len(answer)>=1:
                        writer.writerow([question.strip(), answer.strip()])  # 写入问题和答案

            print("CSV 文件已成功创建！")
            # qa_pairs.append(response)
    # return qa_pairs

# 处理目录中文件
def getDocument(base_directory):

    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.mmd'):
                file_path = os.path.join(root, file)
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 分割文本并生成问答对
                chunks = text_splitter.split_text(content)
                bornQA_repair(chunks)

if __name__ == "__main__":
    base_directory = "./data"  # 替换为实际目录路径
    getDocument(base_directory)
