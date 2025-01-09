from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatBaichuan
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from py2neo import Graph
import os
from dotenv import load_dotenv

from CropGPT.prompt import PARSE_TOOLS_PROMPT_TPL

'''
用来存放一些工具函数，
1. get_embeddings_model：向量化模型，定义了一个函数 `get_embeddings_model()`，根据环境变量选择并返回不同的嵌入模型。
2. get_llm_model：对话大模型，定义了一个函数 `get_llm_model()`，根据环境变量选择并返回不同的对话大模型。
3. structured_output_parser：结构化输出解析器，定义了一个函数 `structured_output_parser(response_schemas)`，生成一个用于解析结构化输出的文本模板。
4. replace_token_in_string：字符串替换，定义了一个函数 `replace_token_in_string(string, slots)`，用于在字符串中替换特定的标记。
5. get_neo4j_conn：Neo4j 连接，定义了一个函数 `get_neo4j_conn()`，用于获取 Neo4j 数据库的连接。
'''


load_dotenv()

# 向量化模型
def get_embeddings_model():
    model_map = {
        # 创建一个字典 model_map，用于存储不同类型的嵌入模型。
        'openai': OpenAIEmbeddings(
            model = os.getenv('OPENAI_EMBEDDINGS_MODEL')
        ),
        'baichuanai': BaichuanTextEmbeddings(),
        # 'bge-large': HuggingFaceBgeEmbeddings(
        #     model_name="BAAI/bge-large-zh",
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        # 'Xinference':XinferenceEmbeddings(server_url="http://127.0.0.1:9997", model_uid="custom-bge-m3")
    }
    return model_map.get(os.getenv('EMBEDDINGS_MODEL'))

# 对话大模型
def get_llm_model():
    model_map = {
        'openai': ChatOpenAI(
            # service = os.getenv('OPENAI_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
        ),
        'baichuan': ChatBaichuan(
            model = os.getenv('BAICHUAN_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
        ),
        'zhipuai':ChatZhipuAI(
            model = os.getenv('ZHIPUAI_LLM_MODEL'),
            temperture = os.getenv('TEMPERTURE'),
        )
#        'qwen':ChatTongyi(
#            service = os.getenv('QWEN_LLM_MODEL'),
#            temperture = os.getenv('TEMPERTURE'),
#        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))


def structured_output_parser(response_schemas):
    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'
    return text


def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)
    return string


def get_neo4j_conn():
    return Graph(
        os.getenv('NEO4J_URI'), 
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

def translate_to_abbreviation(self, region_chinese, phenotype_chinese):
        """
        将中文地区名和表型转换为英文缩写。
        参数:
        region_chinese (str): 中文地区名。
        phenotype_chinese (str): 中文表型名。
        """
        region_abbreviations = {
            "河北": "HeB",
            "吉林": "JL",
            "辽宁": "LN",
            "北京": "BJ",
            "河南": "HN",
        }
        phenotype_abbreviations = {
            "开花期": "DTT",
            "株高": "PH",
            "穗重": "EW",
        }
        region = region_abbreviations.get(region_chinese, None)
        phenotype = phenotype_abbreviations.get(phenotype_chinese, None)
        return region, phenotype

def extract_dna_sequence(self, query):
        """
        从文件中提取DNA序列。
        """
        with open('../example.txt', 'r') as file:
            content = file.read()
        return content

def parse_tools(self, tools, query):
        """
        解析工具描述并选择合适的工具。
        """
        prompt = PromptTemplate.from_template(PARSE_TOOLS_PROMPT_TPL)
        llm_chain = LLMChain(
            llm=self.chat,
            prompt=prompt,
            verbose=True
        )
        # 拼接工具描述参数
        tools_description = ''
        for tool in tools:
            tools_description += tool.name + ':' + tool.description + '\n'
        result = llm_chain.invoke({'tools_description': tools_description, 'query': query})
        # 解析工具函数
        for tool in tools:
            if tool.name == result['text']:
                return tool
        return tools[0]