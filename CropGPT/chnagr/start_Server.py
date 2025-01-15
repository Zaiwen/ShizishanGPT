
from .tool import excute
import re
from datetime import datetime
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .process import loadFile,embeddingText
from prompt import RETRIVAL_PROMPT_TPL
from utils import get_llm_model,get_embeddings_model
import agent
vdb = Chroma(
        embedding_function=get_embeddings_model(),
        persist_directory=os.path.join(os.path.dirname(__file__), "./tempDB/vdb")
)

# 查询向量化库
def queryTemp_Vdb(query):
    # 捕获chn网站
    excute(query)

    loadFile()
    embeddingText()
    # 执行查询
    results = vdb.similarity_search_with_relevance_scores(query, k=3)
    query_result = [doc[0].page_content for doc in results if doc[1]>0.4]
    res = extract_info(query_result)
    print("提取结果",res)
    # extract_info(res)

    if res:
        origin = "\n\n知识来源，中国农业科技信息网:\n"
        agent._flag = True
        for item in res:
            if(origin.__contains__(item['标题'])):
                continue
            origin += '标题:' + item['标题'] + "\n"
            origin += '日期:' + item['日期'] + "\n"
            origin += '链接:' + item['链接'] + "\n"
        agent._origin = origin

    prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
    retrival_chain = LLMChain(
        llm = get_llm_model(),
        prompt = prompt,
        verbose = True
    )
    inputs = {
        'query': query,
        'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
    }
    return retrival_chain.invoke(inputs)['text']




# 提取信息的函数
def extract_info(text):
    # 正则表达式模式
    pattern = r"标题: (.*?)\n日期: (.*?)\n链接: (.*?)\n内容:"
    res = []
    for item in text:
        match = re.search(pattern, item)
        if match:
            res.append({
                '标题': match.group(1).strip(),
                '日期': match.group(2).strip(),
                '链接': match.group(3).strip()
            })
    return res

# # 遍历数据列表，提取每一项的信息
# extracted_info = [extract_info(item) for item in data]


if __name__ == "__main__":
    excute("玉米心叶为什么扭曲")
    res = queryTemp_Vdb("玉米心叶为什么扭曲")
    print(res)