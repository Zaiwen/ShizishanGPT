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
#prompt_str = """
##01 你是数据处理专家。
##02 你需要根据用户信息，提取5个左右的问答对，记住问题和答案都来源于用户信息，请尽量不要自行编造问答对！
##03 问题要宽泛一点，不要针对某一具体文献来提问。避免提及具体的书名、作者、出版信息等。例如，“园林绿化苗木繁育时需要注意哪些方面？”是合适的，而“这本书的出版信息中提到了哪些内容”则不合适。
##04 答案要全面，多使用我的信息，内容要更丰富，并尽量涵盖广泛的应用场景和实际情况。
##05 你必须根据我的对示例格式来生成：
##06 示例格式:
#用户消息:"据了解干旱能够消耗玉米的水分、光照太强能够损失玉米的组织，从而影响玉米生长。"
#输出:
#1、干旱如何影响玉米的生长？
#答:干旱能够消耗玉米的水分。
#2、光照对玉米生长的影响是什么？
#答:光照太强会损害玉米组织，光照需要事宜。
#
##07 真实用户消息如下\n
#用户信息:{User_Msg}
#输出:
#"""
prompt_str = """
#01 你是农业领域论文的问答生成专家。
#02 你的任务是根据用户提供的论文或论文片段，从中提取5个左右的与农业技术、实验方法等实际应用相关的问答对。
#03 问题应聚焦于农业技术细节、实验方法、研究结论等领域，避免提问关于论文格式、章节位置或文献引用等方面的问题。例如：
- 合适的问题：“固定化脂肪酶在生物柴油制备中起什么作用？”
- 不合适的问题：“参考文献位于论文的哪个部分？”、“本研究的目的是什么？”、“论文的主要内容有哪些？”
#04 答案要全面，多使用用户提供的信息，避免凭空编造，确保问题和答案直接来源于用户信息。
#05 生成的问题应涵盖农业领域实验方法、技术流程、研究结果和实际应用，不涉及论文的章节、格式或索引。
#06 你必须根据示例格式生成问答对：
示例格式:
用户消息:"在实验中，五味子油与甲醇按一定摩尔比混合，加入固定化脂肪酶作为催化剂。反应温度设定为50℃，反应时间为6小时。"
输出:
1、五味子油制备生物柴油的实验方法是什么？
答:五味子油与甲醇按摩尔比混合，使用固定化脂肪酶催化，50℃反应6小时。
2、固定化脂肪酶在反应中起到什么作用？
答:固定化脂肪酶作为催化剂，加速五味子油和甲醇的酯交换反应。

#07 真实用户消息如下
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

# 处理所有文本块并将结果追加到 JSON 文件
def bornQA_repair(chunks, json_file_path):
    prompt = PromptTemplate.from_template(prompt_str)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    # 如果 JSON 文件不存在，创建一个新的文件并初始化为空列表
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # 打开现有的 JSON 文件并读取内容
    with open(json_file_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    for chunk in chunks:
        response = process_chunk(chunk, chain)
        if response:
            qa_pairs = re.findall(qa_pattern, response)
            qa_pairs = [(re.sub(r'^\d+、', '', question).strip(), answer.strip()) for question, answer in qa_pairs]

            # 将问答对按格式追加到 JSON 文件中
            for question, answer in qa_pairs:
                if len(question) >= 1 and len(answer) >= 1:
                    output_entry = {
                        "instruction": question,
                        "input": '',
                        "output": answer
                    }
                    existing_data.append(output_entry)

            print("JSON 文件已成功更新！")

    # 将更新后的数据保存回 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

# 处理目录中文件
def getDocument(base_directory, json_file_path):
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.mmd'):
                file_path = os.path.join(root, file)
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 分割文本并生成问答对
                chunks = text_splitter.split_text(content)
                bornQA_repair(chunks, json_file_path)

if __name__ == "__main__":
    base_directory = "./data"  # 替换为实际目录路径
    json_file_path = 'q_a4.json'  # JSON 文件路径
    getDocument(base_directory, json_file_path)