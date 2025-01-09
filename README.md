# 狮山GPT模板文档使用说明
- 12_28 更新: 高度抽离模型调用逻辑，并进一步拆分所有常量（新增config目录），对gradio使用进行简化，常量类得以单独存放。在Service中新增对于DNA链的多个预调用逻辑。整理、优化了AgentHandler中的代码，分离静态方法和实例方法，新增多个CropGPT方法。
## 1. 目录结构
```shell
ModelGPT
├── app.py
├── static
│   └── ...
└── service
    ├── config
    │   ├── GradioConstants.py
    │   ├── Property.py
    │   └── Template.py
    ├── data
    │   └── ...
    ├── other
    │   ├── data_process.py
    │   └── onnx_model.py
    ├── Property.py
    ├── Prompt.py
    ├── Service.py
    ├── AgentHandler.py
    └── Utils.py
```
- app.py：Flask 启动文件，包含前端页面渲染和后端接口调用逻辑
- static：前端静态资源文件夹
- service：后端服务模块
  - config：全局配置文件夹
    - GradioConstants.py：Gradio配置常量
    - Property.py：系统全局配置
    - Template.py：模板配置常量
  - data：数据处理文件夹
  - other：其他模块
  - config.py：配置文件
  - prompt.py：模型提示词
  - Service.py：后端服务逻辑
  - ServiceImpl.py：后端服务实现
  - utils.py：工具函数
- other：其他模块
  - data_process.py：数据处理函数
  - onnx_model.py：ONNX模型调用

在使用本模板文档时，可以根据实际需求修改目录结构，添加新的模块或文件。

## 2. 构建规范
在搭建项目时，建议使用如下思想，以保持代码的规范性和可维护性。
    
- **封装**：将不同功能的代码封装在不同的模块中。这意味着应该根据功能将逻辑细分为清晰的模块或类，以减少代码耦合并提高代码复用性。
使用接口和抽象类来定义模块之间的通信契约。
- **抽象**：将通用的配置信息、工具函数等抽取到配置文件或工具函数模块中，以便后续维护。区分不同的配置环境（开发、测试、生产）。
定义清晰的API接口，以确保系统内各部分可以独立开发和测试。
- **分层架构**：隔离业务逻辑和数据访问逻辑（模型调用），使其更易于测试和维护。
- **单一责任原则**：每个模块或类应仅负责一个功能或概念。避免在单一模块中混合多种职责，以使代码更易于理解与测试。
- **文档和注释**：在代码中添加适当的文档和注释，尤其是对于复杂逻辑。在适当位置添加函数注释说明参数、返回值和方法用途。

## 3.代码功能
`Service.py`

模板的起始文件为Service.py中，对于DNA识别处理的额外逻辑在方法agriculture_bot中额外处理，顶层接口方法为query，这是最终也是宏观的调用方法，
如果有其他额外添加的业务逻辑应在Service.py中添加（**Service相当于SSM架构中一个业务类的接口**），并最终指向query方法。

对于关键方法`query`的详细解释：
```python
# 定义了工具列表，以及实现类Handler中对应指向的方法
tools = [...] 
# 拉取预定义的模板，并补充自定义的提示词
prompt = hub.pull(...)
prompt.template = `...`
# AgentExecutor是最终的执行类，将所有需要调整的参数传入
agent_executor = AgentExecutor.from_agent_and_tools(...)
# invoke方法调用后，会开始根据tools中的工具列表，依次调用思考、再调用(chain)
agent_executor.invoke(...)
```
`AgentHandler.py`

AgentHandler类中实现了具体的模型调用逻辑，其与Service.py中的query方法相对应（二者相当于SSM中Service层和ServiceImpl层，但没有接口实现关系） 

对于关键方法的解析：`fine_qwen`
```python
# 1.阶段一：查询与查询问题相关的文件名
# 构建提示词和模板对象（在config中抽出配置）
prompt = PromptTemplate.from_template(...)
# 创建 LLMChain 实例
llm_chain = LLMChain(...)
# 创建输入字典
inputs_01 = {...}
# 2.阶段二：生成最终回答
# 构建提示词和模板对象
prompt01 = PromptTemplate.from_template(...)
# 创建 LLMChain 实例
fine_chain = LLMChain(...)
# 查询结果
return fine_chain.invoke(...)
```
> 注意：该方法分为两个阶段处理查询，调用两次模型，第一次查询与查询问题相关的文件名，
> 第二次生成最终回答，这这样做的目的是先找到相关的文件，再根据文件内容生成更准确的回答。
 
`generic_func_improve`
```python
# 1.阶段一：查询与用户问题最相似的文件名  
# 读取指定文件夹中的文件名列表  
filenames = read_filenames(folder_path)  
# 根据用户问题，计算文件名相似度并返回最相似的文件  
top_filenames = find_top_n_similar_filenames(query, filenames)  
# 添加兜底文件和微调模型文件到候选文件名列表  
top_filenames.append(...)  
# 2.阶段二：基于文件内容生成回答  
# 读取每个文件的内容  
content = read_file_content(...)  
# 调用语言模型，根据用户问题和文件内容生成回答  
answer = get_model_answer(query, content)  
# 在所有生成的回答中选择与用户问题最相关的答案  
return find_most_similar_answer(query, answers)
```
> 注意：本方法通过两步操作处理一般查询，先找到最相关的文件，再根据文件内容生成回答。
> 这种方式能提高回答的相关性，同时通过相似度计算确保结果的准确性。

`retrival_func_improve`
```python
# 根据用户输入问题，调用 `create_origin_query` 方法生成多个相关查询  
queries = create_origin_query(query)  
# 对生成的查询列表调用 `create_retrieve_documents` 方法，检索文档  
data = create_retrieve_documents(queries)  
# 构建提示词模板和输入字典  
prompt = PromptTemplate.from_template(...)  
inputs = {...}  
# 创建 LLMChain 实例，根据查询结果生成最终答案  
retrival_chain = LLMChain(...)  
return retrival_chain.invoke(inputs)['text']
```
> 该方法相比于 retrival_func 增加了生成相关查询的阶段，适合处理用户问题较模糊或需要精细化处理的场景。
> 通过逐步优化查询内容和检索结果，提高了回答的准确性，但由于要多次处理，因此速度略慢

`search_func`

该方法区别仅在于input输入字典的参数不同，直接附带搜索引擎可以直接调起网页搜索

`graph_func`
```python
# 构建 NER（命名实体识别）提示词模板  
ner_prompt = PromptTemplate(...)  
# 创建 LLMChain 实例，执行命名实体识别任务  
ner_chain = LLMChain(...)  
# 调用链式方法提取关键实体  
result = ner_chain.invoke({'query': query})['text']  
# 使用 StructuredOutputParser 解析 NER 结果，提取结构化数据  
ner_result = output_parser.parse(result)  
# 生成图数据库查询  
# 根据 NER 结果填充模板，生成 Cypher 查询语句  
graph_templates = [...]  # 动态生成与实体相关的 Cypher 查询  
# 使用向量化后的图数据库筛选最相关的模板  
graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)  
# 执行查询并生成答案  
# 遍历最相关的图模板，执行 Cypher 查询语句，获取查询结果  
for document in graph_documents_filter:  
    result = neo4j_conn.run(cypher).data()  
    query_result.append(...)  # 格式化查询结果  
# 构建提示词模板  
prompt = PromptTemplate.from_template(...)  
# 创建 LLMChain 实例，根据图查询结果生成总结回答  
graph_chain = LLMChain(...)  
inputs = {...}  
# 返回最终回答  
return graph_chain.invoke(inputs)['text']
```
> 注意：本方法通过命名实体识别结合图数据库查询，实现复杂关系数据的检索和回答。适用于回答包含多实体、关系结构化问题的场景，但对图数据库的数据质量和模型识别能力有较高依赖。

`predict_crop_phenotype`
```python
# 调用 `parse_query` 方法提取用户查询中的地区、表型和文件名  
region, phenotype, dna_sequence_file = parse_query  
# 根据解析结果生成调用外部预测脚本的命令  
command = "python ./corn_demo/predict.py ./dna_sequence_file --diquname region --name phenotype"
```
> 注意：该方法解析用户输入后，通过外部作物预测脚本完成作物表型预测任务。，对外部脚本和数据的准确性有较高的要求

`Propety`

本类存储系统范围内的参数，如接口地址、模型路径等

本模板在`Service.py`中搭建类似接口的方法，在[`AgentHandler.py`](service/AgentHandler.py)中实现具体的模型调用逻辑，建议将模型调用逻辑封装在`AgentHandler.py`中，以保持`Service.py`的简洁性。

调用模型的路径、模型名称等信息作为常量抽取存放在[`config.py`](service/config/Property.py)中，

### 使用Gradio构建前端页面
若使用gradio搭建前端页面，则无需自定义后端服务器，直接运行[`Service.py`](service/Service.py)方法即可启动服务。

gradio启动搭建方式有多种，本模板使用`chatChatInterface`构建对话界面，如有其他需求参考[Gradio官方文档](https://gradio.app/docs)

gradio搭建前端页面的自定义性较差，如需更加复杂或高度自定义，建议自定义前端页面并调用后端接口，见[自定义前端页面](#自定义前端页面)。

### 自定义前端页面
自定义前端页面通常使用html,css搭配js或前端框架，如Vue、React等，通过调用后端接口实现数据交互。

本模板直接通过js搭建了简单的接口，具体见[`static/index.html`](static/index.html)中引入的js文件。

### 自定义后端服务器
若需要自己构建后端服务器，则在[`app.py`](./app.py)中添加自定义的后端接口，如需更加复杂的逻辑处理，请自定义其他模块实现并引入，以保持app.py的简洁性并解除耦合。

本模板使用python-Flask搭建后端服务，直接运行app.py可启动实例，如有其他需求参考[Flask官方文档](https://flask.palletsprojects.com/en/2.0.x/)

### 提示词
由于提示词数量较多、篇幅较长，建议将提示词放在[`prompt.py`](service/Prompt.py)中，以便后续维护。

### 配置文件
配置文件[`config.py`](service/config/Property.py)中存放了一些常用的配置信息，如模型路径、端口号等，这些信息和逻辑无关，通常进行抽取

### 工具函数
对于一些和业务逻辑无关的工具函数，建议放在[`utils.py`](service/Utils.py)中，有需要直接引入即可。

### 数据处理
数据处理的参考函数位于[`other/data_process.py`](service/other/data_process.py)中，处理后的数据，如搭建的向量数据库建议放在在`data`文件夹中。

### 阅读文档和方法的注释
在关键文档中，于头部编写了大概的注释，以此快速浏览文件内容
在复杂的业务方法下，编写了方法的注释，以此快速了解方法的功能和入参