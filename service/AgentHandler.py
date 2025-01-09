import subprocess
import os.path
from langchain.vectorstores.chroma import Chroma
import re
from fuzzywuzzy import fuzz
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

from CropGPT.config import CROP_GRAPH_TEMPLATE
from Utils import *
from ModelGPT.service.config.Property import *
from langchain_openai import ChatOpenAI
from Prompt import *
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain, LLMRequestsChain
from langchain.prompts import PromptTemplate, Prompt
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

"""
核心：此文件包含项目的服务层实现。
它包括以下功能：
1. 初始化嵌入和聊天模型。
2. 定义 `Agent` 类，该类处理各种查询类型并与不同的模型和数据库交互。
3. 在 `Agent` 类中实现多个方法以处理特定类型的查询，例如：
   - RAG核心代码：retrival_func_edu，用于检索文档。
   - 微调模型核心代码：fine_qwen，用于调用微调模型。
   - `generic_func` 和 `generic_func_edu`：处理一般查询。
   - `graph_func` 和 `graph_zhu_func`：执行命名实体识别并查询 Neo4j 图数据库。
   - `model_func`：根据 DNA 序列预测作物表型。
   - `search_func`：使用搜索引擎回答查询。
4. 定义 `answer` 函数，作为后端服务器请求的入口点。
"""

# 设置环境变量，指定使用的 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# CrossEncoder类已重写
cross_encoder = CrossEncoder(LOCAL_MODEL_PATH,max_length=512)
# 加载 HuggingFace 预训练模型用于生成嵌入向量
embeddings = HuggingFaceEmbeddings(model_name=MODEL_URL)

chat = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key="EMPTY",
    openai_api_base=OPENAI_API_BASE,
    stop=['<|im_end|>'],
    temperature=0.2
)
# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def get_summary_message(message, history):
    llm = get_llm_model()
    prompt = Prompt.from_template(SUMMARY_PROMPT_TPL)
    # 创建一个 LLMChain 实例，传入语言模型实例、提示对象以及是否启用详细输出的信息
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=os.getenv('VERBOSE'))
    chat_history = ''
    # 遍历历史记录的最后两条问答对。
    for q, a in history[-2:]:
        chat_history += f'问题:{q}, 答案:{a}\n'
    #     调用 LLMChain 的 invoke 方法，传入问题和历史对话记录，返回结果中的文本部分。
    return llm_chain.invoke({'query': message, 'chat_history': chat_history})['text']

def split_text_to_strings(text, max_tokens_per_string=500):
    # 使用分词器对文本进行分词
    tokens = tokenizer.tokenize(text)

    # 计算需要分割成多少个字符串
    num_strings = (len(tokens) + max_tokens_per_string - 1) // max_tokens_per_string

    # 根据需要分割的数量，将token列表分割成多个子列表
    split_tokens = [tokens[i * max_tokens_per_string:(i + 1) * max_tokens_per_string] for i in range(num_strings)]
    for to in split_tokens:
        print("----------------------")
        print(len(to))
        print("----------------------")
    # 将每个token子列表转换回字符串
    split_texts = [tokenizer.convert_tokens_to_string(split_tokens[i]) for i in range(num_strings)]

    return split_texts

def create_origin_query(self, original_query):
    """
    根据原始查询生成一组相关查询。
    拓展问题，从一个问题出发，生成相关的多个问题，构成一个问题列表
    参数: original_query (str): 原始查询。
    返回值：queries (list): 问题查询列表。
    """
    query = original_query
    qa_system_prompt = CREATE_QUERIES
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{question}"),
        ]
    )
    rag_chain = (
            qa_prompt
            | self.chat
            | StrOutputParser()
    )
    question_string = rag_chain.invoke(
        {"question": query}
    )
    lines_list = question_string.splitlines()
    queries = [query] + lines_list
    return queries

def create_retrieve_documents(self, queries):
        """
        根据create_original_query生成的查询列表，检索相关文档，找到答案并筛选排序后返回最相关的答案列表
        改进：如果文档过长（超过 500 Token），调用 split_text_to_strings 将文档分割成小块。
        生成文档对后，通过 CrossEncoder 对文档对进行评分，并按分数排序返回前 5 个文档。
        参数: queries (list): 查询列表。
        """
        # 对查询列表 queries每个查询在向量数据库documents中进行相似度搜索，返回前5个最相关的文档
        retrieved_documents = []
        for i, query in enumerate(queries):
            results = self.documents.similarity_search_with_relevance_scores(query, k=5) # 向量相似度（最高5个）搜索
            docString = [doc[0].page_content for doc in results]
            retrieved_documents.extend(docString)
        # 1.使用集合去重
        unique_documents = list(set(retrieved_documents))
        # 2.对 Token 长度的检查：500Token为界限
        # 若文档超长，则调用split_text_to_strings方法将文档分块，并将分块后的文档添加到processed_documents列表中
        processed_documents = []
        for doc in unique_documents:
            tokens = tokenizer.tokenize(doc)
            if len(tokens) > 500:
                text_list = split_text_to_strings(doc)
                processed_documents.extend(text_list)
            else:
                processed_documents.append(doc)
        # 3.生成文档后，通过 CrossEncoder（已重写） 对文档对进行评分，并按分数降序返回前 5 个文档
        pairs = [[queries[0], doc] for doc in processed_documents]
        scores = cross_encoder.predict(pairs)

        final_queries = [{"score": scores[x], "document": processed_documents[x]} for x in range(len(scores))]
        sorted_list = sorted(final_queries, key=lambda x: x["score"], reverse=True)
        first_ten_elements = sorted_list[:5]
        return first_ten_elements

def parse_query(self, query):
    """
    解析查询字符串，提取地区、表型和文件名。
    """
    # 使用正则表达式匹配地区、表型和文件名
    print("开始正则匹配")
    region_pattern = r"河北|吉林|辽宁|北京|河南"
    phenotype_pattern = r"开花期|株高|穗重"
    region_match = re.search(region_pattern, query)
    phenotype_match = re.search(phenotype_pattern, query)

    # 获取中文地区名和表型
    region_chinese = region_match.group(0) if region_match else None
    phenotype_chinese = phenotype_match.group(0) if phenotype_match else None
    dna_sequence_file = '../example.txt'

    # 将中文地区名和表型转换为英文缩写
    print(region_chinese, phenotype_chinese)
    region, phenotype = self.translate_to_abbreviation(region_chinese, phenotype_chinese)

    if not region or not phenotype or not dna_sequence_file:
        raise ValueError("无法找到对应的地区和表型模型，预测失效")

    return region, phenotype, dna_sequence_file

def execute_command(command):
    """
    执行外部命令并返回结果
    """
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        output = result.stdout.strip()  # 去除可能的首尾空白字符
        query_result = f"预测结果为：{output}"
        print(query_result)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        query_result = f"预测过程中出错：{e.stderr}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        query_result = f"预测过程中发生意外错误：{str(e)}"
    return query_result


class AgentHandler() :
    def __init__(self):
        # 初始化时加载两个 Chroma 数据库：一个是用于存储嵌入向量的数据库，另一个是存储文档的数据库
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './data/bc_db'),
            embedding_function = self.embeddings
        )
        self.documents = Chroma(
            persist_directory = "/home/jiaowu/service/db_500_150_emb_3:7",
            embedding_function= self.embeddings
        )

    def fine_qwen(self, x, query):
        """
        使用微调后的模型回答问题
        """
        prompt = PromptTemplate.from_template(FINE_QWEN)

        llm_chain = LLMChain(
            llm=self.chat,
            prompt=prompt,
            verbose=True
        )

        # 处理返回结果
        source = llm_chain.invoke(query)['text']
        source = source.split("source:")[1].strip()
        with open(f"/home/jiaowu/llama/try/total/{source}", "r") as f:
            content = f.read()
        inputs_01 = {
            'query': query,
            'content': content,
        }
        prompt01 = PromptTemplate.from_template('''
         1.请根据以下检索结果，回答用户问题。
         2.检索结果中没有相关信息时，回复“抱歉我暂时无法回答你的问题，如果你想了解更多请去华中农业大学官网”。
         3.当你被人问起身份时，请记住你来自华中农业大学信息学院，是一个教育大模型,是华中农业大学信息学院开发的。
         4.你主要回答教务相关的问题
         5.你必须拒绝讨论任何关于政治，色情，暴力相关的事件或者人物。
         ----------
         用户问题：{query}                                                                  
         ----------
         检索结果：{content}
         -----------

         输出：
     ''')
        fine_chain = LLMChain(
            llm=self.chat,
            prompt=prompt01,
            verbose=True
        )

        return fine_chain.invoke(inputs_01)['text']

    def generic_func(self, x, query):
        """
        使用通用模型回答问题
        """
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL_1)
        llm_chain = LLMChain(
            llm=self.chat,
            prompt=prompt,
            verbose=True
        )
        return llm_chain.invoke(query)['text']

    def generic_func_improve(self, x, query):
        """
         处理一般查询，读取文件夹中的所有文件名，并找到与用户查询最相似的文件名
         参数: x: 未使用的参数。query (str): 查询字符串。
         """
        def read_filenames(folder_path):
            """
            读取文件夹中的所有文件名。
            参数: folder_path (str): 文件夹路径。
            """
            filenames = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    filenames.append(os.path.join(root, file))
            return filenames

        def find_top_n_similar_filenames(question, filenames, n=3):
            """
            找到与问题最相似的前N个文件名。
            参数:
            question (str): 查询字符串。
            filenames (list): 文件名列表。
            n (int): 返回的相似文件名数量。
            """
            similarities = [(filename, fuzz.ratio(question, filename)) for filename in filenames]
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [filename for filename, _ in similarities[:n]]

        def read_file_content(file_path):
            """
           读取文件内容。
           参数: file_path (str): 文件路径。
           """
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except FileNotFoundError:
                print(f"FileNotFoundError: {file_path} not found.")
                return None

        def get_model_answer(query, content):
            """
            根据查询和内容生成答案。
            参数:
            query (str): 查询字符串。
            content (str): 生成文本内容。
            """
            try:
                prompt = PromptTemplate.from_template('''
                        1.请在给定的文本中找出与问题最相关的答案并进行总结，准确且与问题最相关即可，不要添加额外信息。
                        2.如过无法找到或信息不明确或者上下文信息中没有提及，只返回无，不要反回多余的补充信息。
                        ----------
                        用户问题：{query}                                                                  
                        ----------
                        文本：{content}
                        -----------
                        输出：
                        ''')
                inputs = {
                'query': query,
                'content': "\n".join(content),
                }
                # ipdb.set_trace()
                llm = LLMChain(
                    llm = self.chat,
                    prompt = prompt
                )
                # formatted_prompt = prompt.format(query=inputs['query'], content=inputs['content'])
                ans = llm.invoke(inputs)['text']
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "stack overflow" in str(e):
                    print(f"Error: {e}")
                    return "Error: 堆栈溢出或CUDA内存不足，跳过此答案生成。"
                else:
                    raise e
            return ans

        def find_most_similar_answer(question, answers):
            """
             找到与问题最相似的答案。
             参数:
             question (str): 查询字符串。
             answers (list): 答案列表。
             """
            similarities = [(answer, fuzz.ratio(question, answer)) for answer in answers]
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[0][0]

        def get_fine_tune_name(query):
            """
            获取微调模型的文件名。
            参数: query (str): 查询字符串。
            """
            try:
                prompt = PromptTemplate.from_template(FINE_QWEN)
                llm = LLMChain(
                    llm = self.chat,
                    prompt = prompt
                )
                inputs = {
                'query': query,
                }
                source = llm.invoke(inputs)['text']
                source = source.split("source:")[1].strip()
                source = source.split(".txt")[0] + ".txt"
            except IndexError:
                print("IndexError: 'source:' not found in the decoded output.")
            return "/home/jiaowu/llama/try/total/" + source

        folder_path = "/home/jiaowu/llama/try/total"
        filenames = read_filenames(folder_path)
        top_filenames = find_top_n_similar_filenames(query, filenames)
        top_filenames.append("/home/jiaowu/llama/try/total/本科生手册-华中农业大学概况.txt")
        top_filenames.append("/home/jiaowu/llama/try/total/本科生手册-华中农业大学章程.txt")
        # 先加个循环吧说不定只后就用上了
        fine_tune_name = get_fine_tune_name(query)
        top_filenames.append(fine_tune_name)

        answers = []
        for filename in top_filenames:
            content = read_file_content(filename)
            if content:  # 确保文件内容不为空
                answer = get_model_answer(query, content)
                answers.append(answer)
        return find_most_similar_answer(query, answers)

    def retrival_func(self, x, query):
        """
        直接通过向量数据库进行文档检索
        RAG核心代码,直接根据问题从向量数据库vdb 中检索相关文档。
        	用户问题明确，相关文档质量高时，直接检索文档即可，速度快
        """
        # 1.找出与问题最相似的前5个文档
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=5)
        print(documents)
        # 2.筛选出相似度大于0.5的文档
        query_result = [doc[0].page_content for doc in documents if doc[1] > 0.5]
        # 3.填充提示词，将结果传递给语言模型，生成总结的答案
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        retrival_chain = LLMChain(
            llm=self.chat,
            prompt=prompt,
            verbose=True
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return retrival_chain.invoke(inputs)['text']

    def retrival_func_improve(self, x, query):
        """
        （相比于retrival_func方法）先根据问题生成一系列相关的查询，逐步搜索和优化，最后返回结果
        RAG核心代码，根据用户的查询生成一组相关查询，然后从向量数据库中检索相关文档
        改进：用户问题较模糊或较复杂，文档分布广泛且需要优化时，用于提高检索的准确性，速度较慢
        """
        # 1.（根据用户问题）生成相关查询
        queries = create_origin_query(query)
        print(queries)
        # 2.根据查询列表生成相关问题
        data = create_retrieve_documents(queries)
        query_result = data
        # 3.填充提示词，将结果传递给语言模型，最终生成一个更全面、更精准的答案
        prompt = PromptTemplate.from_template(RETRIVAL_FUNC_EDU)
        inputs = {
            'query': query,
            'query_result': ''.join(query_result) if len(query_result) else '没有查到'
        }
        retrival_chain = LLMChain(
            llm=self.chat,
            prompt=prompt,
            verbose=True
        )
        return retrival_chain.invoke(inputs)['text']

    def search_func(self, x, query):
            """
            使用搜索引擎回答查询。
            """
            prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
            llm_chain = LLMChain(
                llm=self.chat,
                prompt=prompt,
                verbose=True
            )
            llm_request_chain = LLMRequestsChain(
                llm_chain=llm_chain,
                requests_key='query_result'
            )
            inputs = {
                'query': query,
                'url': 'https://www.baidu.com/s?wd=' + query
            }
            return llm_request_chain.invoke(inputs)['output']

    def graph_func(self, x, query):
            """
            通过命名实体识别（NER）识别查询中的关键实体，并根据这些实体生成Cypher查询语句来从Neo4j图数据库中检索信息。
            """
            # 命名实体识别
            response_schemas = [

                ResponseSchema(type='list', name='maize', description='玉米品种名称实体'),
                ResponseSchema(type='list', name='crop', description='作物品种名称实体'),
                ResponseSchema(type='list', name='company', description="""选育单位或公司实体
                    ,例如华中农业大学或者农科院"""),
                ResponseSchema(type='list', name='province', description='地区名称实体'),
                ResponseSchema(type='list', name='disease', description='疾病名称实体'),
                ResponseSchema(type='list', name='symptom', description='症状名称实体'),
                # ResponseSchema(type='list', name='drug', description='药物名称实体')
            ]
            # 创建一个 StructuredOutputParser 实例，用于解析命名实体识别的结果。
            output_parser = StructuredOutputParser(response_schemas=response_schemas)
            format_instructions = structured_output_parser(response_schemas)

            ner_prompt = PromptTemplate(
                template=NER_PROMPT_TPL,
                partial_variables={'format_instructions': format_instructions},
                input_variables=['query']
            )

            ner_chain = LLMChain(
                llm=self.chat,
                prompt=ner_prompt,
                verbose=True
            )

            result = ner_chain.invoke({
                'query': query
            })['text']

            print(result)
            # 解析命名实体识别的结果，并转换成结构化的输出。
            ner_result = output_parser.parse(result)
            # 命名实体识别结果，填充模板
            graph_templates = []
            for key, template in CROP_GRAPH_TEMPLATE.items():
                # 获取模板中的槽位名称。
                slot = template['slots'][0]
                # 获取对应槽位的实体值。
                slot_values = ner_result.get(slot, [])
                for value in slot_values:
                    graph_templates.append({
                        'question': replace_token_in_string(template['question'], [[slot, value]]),
                        'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                        'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                    })

            if not graph_templates:
                return

                # 计算问题相似度，筛选最相关问题
            # 将问题取出
            graph_documents = [
                Document(page_content=template['question'], metadata=template)
                for template in graph_templates
            ]
            # db时向量化后的问题
            db = FAISS.from_documents(graph_documents, self.embeddings)
            graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
            # print(graph_documents_filter)

            # 执行CQL，拿到结果
            query_result = []
            neo4j_conn = get_neo4j_conn()
            # 遍历搜索到的最相似的问题。
            for document in graph_documents_filter:
                question = document[0].page_content
                cypher = document[0].metadata['cypher']
                print(cypher)
                answer = document[0].metadata['answer']
                try:
                    result = neo4j_conn.run(cypher).data()
                    if result and any(value for value in result[0].values()):
                        answer_str = replace_token_in_string(answer, list(result[0].items()))
                        query_result.append(f'问题：{question}\n答案：{answer_str}')
                except:
                    pass

            # 总结答案
            prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
            graph_chain = LLMChain(
                llm=self.chat,
                prompt=prompt,
                verbose=True
            )
            inputs = {
                'query': query,
                'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
            }
            return graph_chain.invoke(inputs)['text']

    def predict_crop_phenotype(self, x, query):
        """
        调用外部作物预测模型进行预测
        region: 地区名
        phenotype: 作物类型
        dna_sequence_file: 文档标识符或文件名
        """
        print(f"\npredict_crop_phenotype方法：query {query}\n")
        prompt = PromptTemplate.from_template(MODEL_PROMPT_TPL)
        model_chain = LLMChain(
            llm=self.chat,
            prompt=prompt,
            verbose=True
        )
        region, phenotype, dna_sequence_file = self.parse_query(query)

        # 构建用于调用外部脚本的命令
        script = r"./corn_demo/predict.py"
        input_file = f"./{dna_sequence_file}"
        command = f"python {script} {input_file} --diquname {region} --name {phenotype}"
        print("启动 predict_crop_phenotype")
        print(f"Executing command: {command}")

        # 执行外部命令并返回结果
        return execute_command(command)





