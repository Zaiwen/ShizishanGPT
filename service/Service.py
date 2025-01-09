
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub
from AgentHandler import AgentHandler, get_summary_message
from ModelGPT.service.Gradio import create_gradio_interface
import re

from ModelGPT.service.config.Property import DNA_FILE_NAME

"""
此文件定义了一个类似接口的 `Agent` 类，用于处理各种查询并调用不同的方法来回答问题。
具体的实现在AgentHandler.py中
使用智能代理，这种代理基于 ReAct（Reasoning + Acting） 框架，它能够在回答问题时：
1.推理出该调用哪个工具。
2.根据工具的结果继续选择最合适的下一个工具。
3.输出最终的回答。
- 此外，本类提供一个 Gradio 聊天界面供用户交互的启动方法。
"""



handler = AgentHandler()
clean = False
dnaFile = DNA_FILE_NAME

class Agent:
    @staticmethod
    def query(self, history):
        """
        顶层查询方法，根据查询类型调用不同的方法
        本方法中历史记录history暂未使用
        """
        tools = [
            Tool(
                name = 'generic_func',
                func = lambda x: handler.generic_func(x, self),
                description = '可以解答非专业的通用领域的知识，例如打招呼，问你是谁等问题',
            ),
            Tool(
                name = 'generic_func_improve',
                func = lambda x: handler.generic_func_improve(x, self),
                description = '''当解答华中农业大学教务相关的问题者华中农业大学的相关信息时调用此方法回答,
                当此方法无法回答时调用retrival_func_improve方法''',
            ),
            Tool(
                name = 'retrival_func_improve',
                func = lambda x: handler.retrival_func_improve(x, self),
                description = '''可以解答华中农业大学教务相关的问题例或者华中农业大学的相关信息，
                                如果此方法无法回答教务方面的问题则调用search_func方法''',
               ),
            Tool(
                name = 'retrival_func',
                func = lambda x: handler.retrival_func(x, self),
                description = '优先使用此工具回答农业领域的问题'
                     '或者当问题中包含英文单词或者是问题中包含土壤改良、滴灌技术、防治蔬菜水果病虫害'
                            '化肥、农药等农业领域常见的专业问题时，也使用该工具回答',
            ),
            Tool(
                name='search_func',
                func=lambda x: handler.search_func(x, self),
                description='其他工具没有正确答案时，最后通过搜索引擎，回答用户问题',
            ),
            # 用于回答水稻种植相关问题
            Tool(
                name = 'graph_func',
                func = lambda x: handler.graph_func(x, self),
                description = '用于回答玉米的生育期、百粒重、栽培要点、穗长、穗位高，特征特征、抗病类型、播种密度、收获时间、播种时间、产量估计、选育单位玉米品种的相关问题'
            ),
            Tool(
                name = 'model_func',
                func = lambda x: handler.predict_crop_phenotype(x, self),
                description = '用于预测表型结果,表型有株高，穗重，开花期三种 ，地区有河北，吉林，辽宁，北京，河南五种，两个条件缺一不可'
                ),

        ]

        # 拉取为 ReAct框架预先设计的标准提示模板，并补充自定义提示
        prompt = hub.pull('hwchase17/react-chat')
        prompt.template = '''你是华中农业大学信息学院开发的教务GPT,
            你有三种方法来华中农业大学教务或者与华中农业大学相关回答问题：
            1. 优先使用generic_func_edu方法来回答
            2. 如果generic_func_edu方法无法回答则使用 retrival_func_edu方法来获取与问题相关的知识。
            3. 如果 retrival_func_edu 方法不能给出完整答案或者回答“抱歉，根据提供的检索结果，我无法回答这个问题”这样的答案，
            尝试用search_func方法回答。

            请按顺序尝试这些方法每个方法只能调用一次，直到问题得到完整的回答。如果所有方法都无法回答，请提示用户提供更多信息。''' + prompt.template
        agent = create_react_agent(llm=handler.chat, tools=tools, prompt=prompt)
        # 创建一个内存对象，用于存储对话历史
        memory = ConversationBufferMemory(memory_key='chat_history')
        # 创建一个代理执行器
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent = agent,
            tools = tools,
            memory = memory,
            handle_parsing_errors = True,
            verbose = True
        )
        res = None
        try:
            res = agent_executor.invoke({"input": self})['output']

        except Exception as e:
            print(e)
            res = "抱歉，暂时无法解答您的问题"
        finally:
            return res

    @staticmethod
    def agriculture_bot(self, message, history, dna_sequence):
        """
        此方法用于处理农业相关的查询，来自CropGPT，并根据输入的消息、历史记录和 DNA 序列执行相应的操作。
        参数:
        - self: 类的实例
        - message: 用户输入的消息
        - history: 对话历史记录
        - dna_sequence: 输入的 DNA 序列
        返回: 根据输入的消息和 DNA 序列返回相应的响应消息
        1. 如果 `clean` 标志为 True，则清理历史记录的最后两条记录。
        2. 如果没有消息且有有效的 DNA 序列，则将 DNA 序列写入文件并设置 `clean` 标志为 True。
        3. 如果有消息且有有效的 DNA 序列，则将 DNA 序列写入文件并调用 `answer` 方法返回响应。
        4. 如果 DNA 序列无效，则返回提示消息并设置 `clean` 标志为 True。
        5. 如果没有消息且没有 DNA 序列，则返回提示消息并设置 `clean` 标志为 True。
        6. 否则，调用 `answer` 方法返回响应。
        """
        global clean
        def write_dna_sequence(dna_sequence):
            with open(dnaFile, "w") as f:
                f.write(dna_sequence)
            print('写入成功！！')

        if clean:
            print("清理历史记录")
            del history[-2:]
        # 如果 message 长度为0，并且 dna_sequence 不为空
        # 并且 dna_sequence 完全匹配正则表达式 ([ATGC]+)（即 dna_sequence 仅包含 A、T、G、C 字符）。
        if not message and dna_sequence and re.fullmatch(r'([ATGC]+)', dna_sequence):
            write_dna_sequence(dna_sequence)
            clean = True
        elif message and dna_sequence and re.fullmatch(r'([ATGC]+)', dna_sequence):
            write_dna_sequence(dna_sequence)
            clean = False
            return self.query(message, history)
        elif dna_sequence and not re.fullmatch(r'([ATGC]+)', dna_sequence):
            clean = True
            return '请输入正确的DNA序列😄'
        elif not message and not dna_sequence:
            clean = True
            return '请输入您的问题😀'
        else:
            clean = False


# 如果要使用gradio，则使用以下代码：
if __name__ == '__main__':
    interface = create_gradio_interface()
    interface.launch(share=True, server_name='0.0.0.0', server_port=7901)