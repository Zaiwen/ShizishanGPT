from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
import json
from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema import AIMessage

from config import *
from prompt import *
from utils import *
import requests
import re
from langchain.chains import ConversationChain,LLMChain
from langchain.prompts import PromptTemplate
from difflib import SequenceMatcher
import ast
import os
import os.path
#糯玉米：口感绵糯，含糖量低，适合减肥的人食用
#糯带皮玉米：方便携带，口感甜蜜，含糖量低
#黄玉米：黄色，含糖量最低，适合减肥的人食用
#黄带皮玉米：方便携带，含糖量低，适合野餐，郊游等情况下选择
#黑糯: 富含膳食纤维，含糖量低，口感糯

count = 0
llm = get_llm_model()
memory = ConversationBufferMemory()

prompt = '''
你是狮子山大模型赋能的玉米搬运机器人，请你根据我的指令，以json形式输出接下来要运行的对应函数和你给我的回复。
你只需要结合场景信息思考并回答一个列表即可，不要回答任何中文。

我们还具有玉米专业知识问答功能，当用户问题与“玉米的生育期、百粒重、栽培要点、穗长、穗位高，特征特征、抗病类型、播种密度、收获时间、播种时间、产量估计、选育单位玉米品种相关”是，你也可以进行回答
你要根据用户需求结合餐厅的玉米种类和其相应特征来判断出用户真正想要的玉米品种，如果用户没有想要任何玉米的倾向，则无须考虑场景信息。
如果用户提出产品不在场景中的玉米种类中则需要回复：{{'crop':‘玉米’,'text':'抱歉，我们这里没有你提到的产品','response':'chat'}}

【输出限制】
你直接输出json即可，从{{开始，以}}结束，不要输出包含```json的开头或结尾
在'crop'键中，输出玉米种类，列表中每个玉米名称都是字符串，用户询问时，输出场景中你认为最合适的玉米，当用户回答需要或者表示肯定的意图时，输出{{history}}中最近的对话中选择的玉米
在'text'键中，用户询问时解释为什么选择该玉米,并询问是否需要拿取，当用户回答需要或者表示肯定的意图时，回答:好的，接下来我会为您拿XX玉米
在'response'键中，用户问需要何种玉米时为chat，当用户回答需要或者表示肯定的意图时为act


【以下是一些具体的多轮对话例子】
我的指令：我需要一些低GI的食物来控制我的血糖，你能帮我吗？\n场景信息:['黑糯'、'黄玉米']
你回复： {{'crop':‘黄玉米’,'text':'当然可以。我们这里有新鲜的黄玉米，它的GI值较低，适合血糖控制。你需要我帮你拿一些吗？','response':'chat'}}
我的指令：好的，请帮我拿一些。\n场景信息:['黑糯'、'黄玉米']
你回复:  {{'crop':‘黄玉米’,'text':'好的，接下来我会为您拿黄玉米','response':'act'}}
我的指令：我正在寻找一些富含膳食纤维的食物，你能推荐吗？ \n场景信息:['黑糯'、'黄玉米']
你回复： {{'crop':‘黄玉米’,'text':'当然可以。我们这里有黑糯玉米，它不仅富含膳食纤维，还含有抗氧化的花青素。你需要我帮你拿一些吗？','response':'chat'}}
我的指令：需要。\n场景信息:['黑糯'、'糯玉米']
你回复:  {{'crop':‘黑糯’,'text':'好的，接下来我会为您拿黑糯玉米','response':'act'}}
我的指令：我想要一些方便携带的玉米，你有什么推荐的嘛？\n场景信息:['黑糯'、'黄玉米']
你回复：{{'crop':‘黄玉米’,'text':'我们这里黄带皮玉米，糯带皮玉米，请问你需要嘛','response':'chat'}}
我的指令：来一些黄带皮玉米吧？\n场景信息:['黄玉米'、'黄带皮玉米']
你回复：{{'crop':‘黄带皮玉米’,'text':'好的，接下来我会帮你拿黄带皮玉米','response':'act'}}


【我现在的指令和场景信息是】
Human:
'''


Answer_right_TPL = """
根据答案判断前一个大模型是否回答了用户问题。
如果回答了，返回true,否则返回false
最终输出只需给出true或者false中的一个

例如:
用户消息:黑糯305的生育期\n答案:91天左右\n输出:true
用户消息:先玉2036的百粒重\n答案:抱歉，暂时无法解答\n输出:false
用户消息:黑糯是什么\n答案:对不起，我未能找到关于黑糯的具体品种信息，它可能需要更详细的描述或者我可能需要检索其他工具。请提供更多信息，以便我能提供准确的帮助。\n输出:false
用户消息:您好\n答案:你好\n输出:true
用户消息:玉米种植中常见的病害有哪些？\n答案:根据现有资料，没有直接提到玉米种植中常见的病害。\n输出:false
用户消息:什么食物富含维生素A\n答案:对不起，我未能找到相关资料\n输出:false
用户消息:你是谁\n答案:我是来自华中农业大学信息学院的智能机器人，你可以称呼我为小智。我可以帮助你解答关于玉米以及其他一些问题，有什么我可以帮助你的吗？\n输出:true

用户消息:{question}
答案:{answer}
输出:
"""

JUDGE_TPL = """ 
根据用户输入，判断用户问题是否为玉米领域的专业性问题\n
如果是玉米领域的专业性问题，比如玉米栽培、病虫害预防等等，请返回true\n
如果是日常生活中的问题，比如有关玉米的饮食、营养价值等等，请返回false\n
你可以结合下面例子，判断用户问题类型\n

例:
用户消息:黑糯305的生育期\n输出:true
用户消息:我正在寻找一些富含膳食纤维的食物，你能推荐吗？\n输出:false
用户消息:我听说玉米对心脏有好处，这是真的吗？ \n输出:false
用户消息:我想了解一下，玉米和玉米制品在营养价值上有什么不同？\n输出:false
用户消息:玉米种植中常见的病害有哪些？ \n输出:true
用户消息:干旱胁迫对玉米生长的影响\n输出:true
用户消息:如何预防玉米纹枯病？\n输出:true
用户消息:我正在寻找一些低卡路里的食物，玉米符合这个要求吗？ \n输出:false
用户消息：华中农业大学最新可以成果?\n 输出:true

用户消息:{question}
输出:
"""
# system_message = SystemMessage(content=prompt)
# human_message = HumanMessagePromptTemplate.from_template("{history}")
# prompt = ChatPromptTemplate(messages=[system_message, human_message])
# prompt_template = PromptTemplate(
#     input_variables=["user_input"],
#     template=prompt
# )

# 判断CropGPT是否正确给出答案
def JudgeCropGPT(query,answer):
    _prompt = PromptTemplate.from_template(Answer_right_TPL)
    _chain = LLMChain(
        llm = llm,
        prompt = _prompt,
        verbose = True
    )
    inputs = {
        "question" : query,
        "answer" : answer
    }
    _res = _chain.invoke(inputs)["text"]
    if _res.__contains__("false"):
        return llm(query+"\n请简要回答").content
    
    return answer

def JudgeWhetherIsProficient(msg):
    _prompt = PromptTemplate.from_template(JUDGE_TPL)
    _chain = LLMChain(
        llm = llm,
        prompt = _prompt,
        verbose = True
    )
    return _chain.invoke({"question":msg})["text"]

memory = ConversationBufferMemory()  # 全部存储
def getanswer(query,senario):
    global count
    global description
    query1=query
    if count==0:
        query=prompt+query+"\n"+senario
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # 开启查看每次prompt内容
    )
    count=count+1
    result=conversation.run(query)
    data=result

    _type = JudgeWhetherIsProficient(query1)

    data = data.replace("{{", "{").replace("}}", "}")
    data_dict = ast.literal_eval(data)

    # 提取 response 的值
    response_type = data_dict.get('response')
    if response_type=="act" or _type.__contains__("false"):
        return result
    else:
        url = "http://211.69.141.168:5003/invoke"
        params = {'question': query1}
        if(count>10):
            print("    清除记录！！！！")
            memory.clear
            count=0
        result1 = requests.get(url, params=params)
        if result1.status_code == 200 :
            print("采用狮子山GPT")
            res_dict = {
                'crop':"None",
                'text': JudgeCropGPT(query1,result1.text),
                'response':'chat'
            }
            
            if(res_dict["text"] == "抱歉，暂时无法解答您的问题"):
                return "出现问题"
            else:
                memory.save_context({"input": query1},
                        {"output": "{res_dict}"})
                # update_memory_with_new_text(result, conversation)
                return "{" + json.dumps(res_dict, ensure_ascii=False) + "}"       
        else:
            print("网络故障")
            return "Internet error"


def update_memory_with_new_text(new_text, model):
    """
    通过再次调用大模型，让它记住新的文本信息并更新 memory。
    
    参数:
    - new_text: 要添加的新的文本内容。
    - model: 用于生成回复和更新记忆的模型对象。
    
    返回:
    - 更新后的 memory。
    """
    
    
    # 构造 prompt，以便大模型知道如何更新记忆
    prompt = f"""

    对于AI的最后一段回复，用户提供了新的text回答，请你进行更新替换:
    {new_text}
    请结合这些信息更新对话历史，并给出一个合适的回复。

    """
    
    # 调用大模型生成新回复，并更新 memory
    response = model.run(prompt)  
    
    return None


   


if __name__ == '__main__':
    print(getanswer("我需要一些低GI的食物来控制我的血糖，你能帮我吗？ "))
    print(getanswer("好的，请帮我拿一些。"))
    print(getanswer("我最近在减肥，你能推荐一些食物吗？ "))
    print(getanswer("你这里有什么食物？"))
    print(getanswer("我听说玉米对心血管健康有好处"))
    print(getanswer("我想要一些方便携带的玉米，你有什么推荐的嘛？"))

    








