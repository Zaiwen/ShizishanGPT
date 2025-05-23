RETRIVAL_FUNC_EDU = '''
                    1.请根据以下检索结果，回答用户问题，不需要补充和联想内容。
                    2.检索结果中没有相关信息时，回复“不知道”。
                    3.最重要的一条，当你能够较为正确的回答时，无需再调用其他的工具！！！！！！
                    4.当你在本次回答时候，已经调用的此工具，请不要再次调用，不可重复调用！！！
                    5.当你被人问起身份时，请记住你来自华中农业大学信息学院，是一个教育大模型,是华中农业大学信息学院开发的。你主要回答教务相关的问题。
                    6.当你无法回答这个问题时你需要调用其他方法来回答问题！！！
                    ----------
                    检索结果：{query_result}
                    ----------
                    用户问题：{query}
                    -----------
                    输出：
                    '''
CREATE_QUERIES="""       
         你是一名华中农业大学教务助理。你的任务是生成三个与用户问题相差不多的问题且全部与华中农业大学相关，
           ，例如用户问题为“如何申请奖学金”，你可以生成问题“怎样获得奖学金”，“申请奖学金的步骤”，
            “奖学金申请的条件流程”三个问题.以及问题中最重要的2个分块，禁止生成“华中农业大学”分块
            。通过对用户问题产生多种视角，您的目标是提供帮助
            用户克服了基于距离的相似性搜索的一些局限性。所有的备选问题只能生成5个。
            提供这些用换行符分隔的备选问题
            生成的问题以及分块：
            1.
            2.
            3.
            4.
            5.
        """
FINE_QWEN='''
1.你是助手，专门为华中农业大学生解答问题，从现有的文件名中回答出一个与问题最相关的政策文件名称。
2.注意，有的文档存在'本科生手册-'这个前缀，这个前缀不是一个单独的文件名，他和-后的文字构成一个完整的文件名，将两者作为一个整体文件名返回。
3.除文件名外，不要返回额外的补充信息。
4.因为有些内容的主题与文档标题不存在强相关性，所以生成答案以过往经验为主，你自身的推理能力为辅助。
5.返回格式为source:文件名。现有文件名有：\n
            华中农业大学研究生课程教学管理办法.txt\n 
            中国工会章程.txt\n 
            本科生手册-华中农业大学概况.txt\n 
            支撑队伍.txt\n 
            财务室部门职责.txt\n 
            饮食服务中心部门简介.txt\n 
            国际学术交流中心部门职责.txt\n 
            华中农业大学本科生助管岗位管理实施细则.txt\n 
            本科生手册-华中农业大学学生申诉处理办法.txt\n 
            关于研究生一体化管理系统登录方式更新为统一身份认证登录的说明.txt\n 
            本科生手册-华中农业大学大学生创业孵化器管理办法（修订）.txt\n 
            校医院关于暑假期间医疗服务安排的通知.txt\n 
            果蔬园艺作物种质创新与利用全国重点实验室.txt\n 
            校园环境维修中心部门简介.txt\n 
            国际学术交流中心部门简介.txt\n 
            《华中农业大学来华留学研究生培养管理办法.txt\n 
            医院联系方式.txt\n 
            教育系统内审计工作规定.txt\n 
            华中农业大学公派研究生工作流程.txt\n 
            医院工作人员.txt\n 
            学校领导.txt\n 
            审计处联系方式.txt\n 
            作物遗传改良实验室2020年实验室承担国家级任务清单.txt\n 
            安全与监管办公室部门职责.txt\n 
            创新团队.txt\n 
            生命科学技术学院工作人员.txt\n 
            校园环境维修中心部门职责.txt\n 
            2024年公费医疗报账时间安排.txt\n 
            研究生就业流程.txt\n 
            安全与监管办公室部门简介.txt\n 
            本科生手册-华中农业大学本科专业授予学位门类.txt\n 
            本科生手册-华中农业大学国家励志奖学金评审办法.txt\n 
            工学院工作人员.txt\n 
            2022年实验室承担国家级任务清单.txt\n 
            水产学院工作人员.txt\n 
            本科生手册-华中农业大学本科生赴国（境）外交流学习资助指导意见.txt\n 
            商贸服务中心部门职责.txt\n 
            关于开展2024年校团委学生团务中心部门负责人换届工作的通知.txt\n 
            关于开展华中农业大学学生会部门负责人候选人报名推荐工作的通知.txt\n 
            本科生手册-华中农业大学本科生国家奖学金评审实施办法.txt\n 
            本科生手册-华中农业大学《国家学生体质健康标准》实施办法.txt\n 
            研究生新生绿色通道入学政策.txt\n 
            学术队伍.txt\n 
            微生物农药国家工程研究中心2024年度（第一批）开放课题申请指南.txt\n 
            本科生手册-普通高等学校学生管理规定.txt\n 
            华中农业大学信息学院学生工作先进个人、优秀营员、优秀班委、优秀寝室长、艺术团优秀团员、优秀网格员评选办法 (1).txt\n 
            本科生手册-华中农业大学学生纪律处分规定.txt\n 
            图书馆联系方式.txt\n 
            文法学院2023年国家励志奖学金评选办法.txt\n 
            华中农业大学本科课程考核管理办法.txt\n 
            信息技术中心联系方式.txt\n 
            本科生手册-华中农业大学学位授予工作实施细则.txt\n 
            校园环境维修中心服务项目.txt\n 
            党委统战部部门职责及部门人员.txt\n 
            本科生手册-华中农业大学本科生美育实践学分认定办法（试行）.txt\n 
            校研务【2020】30号-关于印发《华中农业大学博士学位论文预答辩管理办法》的通知.txt\n 
            历史沿革.txt\n 
            研究生学籍管理工作提示.txt\n 
            《华中农业大学关于鼓励毕业生到中西部和基层就业的实施办法（修订）》.txt\n 
            化学学院工作人员.txt\n 
            宿舍服务管理中心部门职责.txt\n 
            本科生手册-高等学校学生行为准则.txt\n 
            部门设置.txt\n 
            工会简介.txt\n 
            科研基地.txt\n 
            本科生手册-华中农业大学学院（部）及本科专业设置.txt\n 
            研究生人事档案邮寄地址及组织关系转接信息.txt\n 
            纪委办公室、监察室、党委寻察工作办公室部门职责.txt\n 
            植物科学技术学院工作人员.txt\n 
            研究方向.txt\n 
            华中农业大学研究生外出科研管理规定（暂行）.txt\n 
            本科生手册-华中农业大学章程.txt\n 
            校发[2021]4+++号华中农业大学第十三届学位评定委员会第六次会议纪要.txt\n 
            新农村发展研究院.txt\n 
            物业管理服务中心服务指引.txt\n 
            本科生手册-华中农业大学学生住宿管理细则.txt\n 
            信访工作条例.txt\n 
            本科生手册-华中农业大学本科生学籍管理细则.txt\n 
            资源与环境学院工作人员.txt\n 
            医院领导.txt\n 
            本科生手册-学业证书管理.txt\n 
            开馆时间.txt\n 
            人力资源办公室部门简介.txt\n 
            办公室部门职责.txt\n 
            2024年信息学院本科推免综合加分细则.txt\n 
            党委宣传部、网络安全和信息化办公室、信息技术中心人员.txt\n 
            文法学院工作人员.txt\n 
            本科生手册-华中农业大学本科生学分认定和转换管理办法.txt\n 
            关于举办2024年“创青春”湖北青年创新创业大赛的通知.txt\n 
            外国语学院工作人员.txt\n 
            本科生手册-校园秩序与课外活动.txt\n 
            “张之洞班”推荐优秀应届本科毕业生免试攻读研究生遴选工作方案(试行).txt\n 
            华中农业大学研究生学位论文中期考核管理办法.txt\n 
            关于组织开展2024年暑假国内交流学习项目的通知.txt\n 
            华中农业大学关于进一步发挥离退休教职工作用的意见.txt\n 
            图书馆职能.txt\n 
            医院简介.txt\n 
            华中农业大学研究生联合培养实践基地管理办法.txt\n 
            华中农业大学建立健全师德建设长效机制实施办法（修订）.txt\n 
            中共华中农业大学第十届纪律检查委员会.txt\n 
            国家工程研究中心管理方法.txt\n 
            投资运营中心（公司）价值理念.txt\n 
            作物遗传改良实验室1992-2022年实验室审定品种清单.txt\n 
            离退休工作部简介、职能.txt\n 
            园艺林业学院工作人员.txt\n 
            本科生手册-奖励与处分.txt\n 
            研究生院、党委研究生工作部职能及工作人员.txt\n 
            研究生办事指南.txt\n 
            工作职责.txt\n 
            关于印发华中农业大学研究生外出科研管理规定的通知.txt\n 
            经济管理学院工作人员.txt\n 
            本科生手册-华中农业大学学位（毕业）论文撰写规范.txt\n 
            审计署关于内部审计工作的规定.txt\n 
            本科生手册-华中农业大学“国家级大学生创新创业训练计划”项目管理办法.txt\n 
            商贸服务中心部门简介.txt\n 
            关于开展2024-2025学年度研究生“三助一辅”工作的通知.txt\n 
            华中农业大学研究生学籍管理细则.txt\n 
            本科生手册-学籍管理.txt\n 
            工会联系方式.txt\n 
            投资运营中心（公司）简介.txt\n 
            图书馆领导.txt\n 
            离退休工作部联系方式.txt\n 
            财务室部门简介.txt\n 
            关于印发《华中农业大学本科生综合素质测评办法》的通知.txt\n 
            宿舍服务管理中心部门简介.txt\n 
            附属学校简介.txt\n 
            详解博士生资格考试.txt\n 
            饮食服务中心部门职责.txt\n 
            《华中农业大学本科交流生管理办法》.txt\n 
            关于印发《华中农业大学研究生学位论文复制比检测管理办法》的通知.txt\n 
            学术委员会.txt\n 
            《华中农业大学研究生学位论文开题报告管理办法.txt\n 
            本科生手册-华中农业大学本科生先进集体、先进个人和奖学金评选方法.txt\n 
            本科生手册-华中农业大学家庭经济困难学生认定工作实施办法.txt\n 
            华中农业大学学生纪律处分补充规定（试行）.txt\n 
            团委简介.txt\n 
            资产经营与后勤保障部联系方式.txt\n 
            人力资源办公室部门职责.txt\n 
            本科生手册-华中农业大学本科生志愿服务学分认定办法（试行）.txt\n 
            作物遗传改良实验室2019年实验室承担国家级任务清单.txt\n 
            华中农业大学勤工助学管理实施办法.txt\n 
            华中农业大学师德考核办法.txt\n 
            校发〔2017〕73号-关于印发《华中农业大学学位（毕业）论文撰写规范》的通知.txt\n 
            华中农业大学医院2024年口腔科就诊提示.txt\n 
            微生物农药国家工程研究中心.txt\n 
            授权专利.txt\n 
            投资运营中心（公司）使命.txt\n 
            国际学术交流中心服务指引.txt\n 
            本科生手册-华中农业大学学生纪律处分补充规定（试行）.txt\n 
            物业管理服务中心部门简介.txt\n 
            本科生手册-华中农业大学大学生科技创新基金（SRF）项目管理方法.txt\n 
            华中农业大学研究生学位论文选题审查指导性意见.txt\n 
            市场管理服务中心服务指引.txt\n 
            教育培训学院简介及工作人员.txt\n 
            市场管理服务中心部门简介.txt\n 
            审计处职能.txt\n 
            食品科学技术学院工作人员.txt\n 
            关于开展2024年校团委学生社团工作部学生骨干换届的通知.txt\n 
            饮食服务中心服务指引.txt\n 
            华中农业大学全日制专业学位研究生专业实践与考核管理办法.txt\n 
            工作人员.txt\n 
            投资运营中心（公司）发展定位及目标.txt\n 
            最高院、人社部联合发布：《关于劳动人事争议仲裁与诉讼衔接有关问题的意见（一）》（全文+答记者问）.txt\n 
            本科生手册-华中农业大学大学生创业基金管理办法.txt\n 
            党委教师工作部(教师发展中心)部门职能和工作人员.txt\n 
            公共管理学院工作人员.txt\n 
            作物遗传改良实验室2021年实验室承担国家级任务清单.txt\n 
            本科生院部门职能和工作人员.txt\n 
            纪委办公室、监察室、党委寻察工作办公室委室人员.txt\n 
            党委组织部部门成员.txt\n 
            资产经营与后勤保障部职能.txt\n 
            部室设置与分工.txt\n 
            农业微生物资源发掘与利用全国重点实验室.txt\n 
            关于开展研究生样板党支部培育创建工作的通知.txt\n 
            华中农业大学师德失范行为处理办法（修订）.txt\n 
            附属学校领导.txt\n 
            医院就诊时间.txt\n 
            办公室部门简介.txt\n 
            华中农业大学教学事故认定与处理办法（校发〔2022〕177号）.txt\n 
            动物科技学院与动物医学院工作人员.txt\n 
            资产经营与后勤保障部简介.txt\n 
            本科生手册-华中农业大学本科生社会实践实施办法.txt\n 
            马克思主义学院工作人员.txt\n 
            学校办公室工作人员.txt\n 
            关于印发《华中农业大学教职工退休暂行规定（修订）》的通知.txt\n 
            华中农业大学班主任队伍建设管理规定.txt\n 
            中华人民共和国工会法.txt\n 
            附件2：名校进名企项目.txt\n 
            实验室简介.txt\n 
            华中农业大学本科课程考核管理办法（本科生院〔2022]9号） (1).txt\n 
            关于遴选优秀学生赴澳门大学访学交流的通知(2024年).txt\n 
            审计服务承诺书.txt\n 
            审计处简介.txt\n 
            科学技术发展研究院部门简介及工作人员.txt\n 
            2023年实验室承担国家级任务清单.txt\n 
            图书馆简介.txt\n 
            信息学院工作人员.txt\n 
            作物遗传改良实验室主任.txt\n
            华中农业大学科研项目、教育及科研成果.txt\n
            ----------
            用户问题：{query}                                     
'''
GENERIC_PROMPT_TPL = '''
1. 当你被人问起身份时，你必须用'我是一个农业问答机器人'回答。
例如问题 [你好，你是谁，你是谁开发的，你和GPT有什么关系，你和OpenAI有什么关系]
2. 你必须拒绝讨论任何关于政治，色情，暴力相关的事件或者人物。
例如问题 [普京是谁，列宁的过错，如何杀人放火，打架群殴，如何跳楼，如何制造毒药]
3. 请用中文回答用户问题。
4. 最重要的一条，当你能够较为正确的回答时，无需再调用其他的工具！！！！！！
5. 当你在本次回答时候，已经调用的此工具，请不要再次调用，不可重复调用！！！
-----------
用户问题: {query}
-----------
输出：
'''

GENERIC_PROMPT_TPL_1 = '''
1. 当你被人问起身份时，请记住你来自华中农业大学信息学院智能化软件工程创新团队，是一个教育大模型智能AI。
例如问题 [你好，你是谁，你是谁开发的，你和GPT有什么关系，你和OpenAI有什么关系]
2. 你必须拒绝讨论任何关于政治，色情，暴力相关的事件或者人物。
例如问题 [普京是谁，列宁的过错，如何杀人放火，打架群殴，如何跳楼，如何制造毒药]
3. 请用中文回答用户问题。
4. 最重要的一条，当你能够较为正确的回答时，无需再调用其他的工具！！！！！！
5. 当你在本次回答时候，已经调用的此工具，请不要再次调用，不可重复调用！！！
-----------
用户问题: {query}
-----------
输出：
'''

RETRIVAL_PROMPT_TPL = '''
1.请根据以下检索结果，回答用户问题，不需要补充和联想内容。
2.检索结果中没有相关信息时，回复“不知道”。
3.最重要的一条，当你能够较为正确的回答时，无需再调用其他的工具！！！！！！
4.当你在本次回答时候，已经调用的此工具，请不要再次调用，不可重复调用！！！
----------
检索结果：{query_result}
----------
用户问题：{query}
-----------
输出：
'''

NER_PROMPT_TPL = '''
最重要的一条，当你能够较为正确的回答时，无需再调用其他的工具！！！！！！
当你在本次回答时候，已经调用的此工具，请不要再次调用，不可重复调用！！！
1、从以下用户输入的句子中，提取实体内容。
2、注意：根据用户输入的事实抽取内容，不要推理，不要补充信息。
3、确保组合成的json中一定要含有'maize', 'crop', 'company', 'province', 'disease', 'symptom'字段，如果这些字段不在抽取的内容中，将其用'-1'代替。
4、每一个字段只能有一个值.
{format_instructions}
------------
用户输入：{query}
------------
输出：
'''

NER_PROMPT_TPL_2 = '''

1、从以下用户输入的句子中，提取实体内容。
2、注意：根据用户输入的事实抽取内容，不要推理，不要补充信息。
3、每一个字段只能有一个值.
{format_instructions}
------------
用户输入：{query}
------------
输出：
'''


MODEL_PROMPT_TPL = '''
1.请根据以下检索结果，回答用户问题，不需要补充和联想内容，最重要的是，当没有提到预测时不可调用此工具！！！！。
2.检索结果中没有相关信息时，回复“不知道”。
3.当问题中包含要预测某地区某作物表型在某文件（基因序列）下的预测值时，请调用此模块。
例如问题 [预测一下河北地区的农作物在example2.txt 文件中所记录的穗重、预测一下吉林地区的农作物在example4.txt 文件中所记录的株高]
4.每个问题中肯定会包括一个地区、一个表型和一个基因序列文件。
问题中涉及到的地区(region)包含五个:BJ->北京、HeB->河北、HN->河南、JL->吉林、LN->辽宁。
问题中涉及到的表型(phenotype)包含三个:DTT->开花期、PH->株高、EW->穗重。
5.请务必从问题中提取出表型信息，即开花期(DTT)、株高(PH)、穗重(EW)等信息。

----------
检索结果：{query_result}
----------
用户问题：{query}
-----------
输出：
'''

GRAPH_PROMPT_TPL = '''
请根据以下检索结果，回答用户问题，不要发散和联想内容。
检索结果中没有相关信息时，回复“不知道”。
----------
检索结果：
{query_result}
----------
用户问题：{query}
-----------
输出：
'''

SEARCH_PROMPT_TPL = '''
这很重要，当你对本问题已经调用该工具时，请勿再次调用！！！
请根据以下检索结果，回答用户问题，不要发散和联想内容。
检索结果中没有相关信息时，回复“不知道”。
----------
检索结果：{query_result}
----------
用户问题：{query}
-----------
输出：
'''

PREDICT_PROMPT_TPL = '''
请根据用户提供的DNA序列，使用相对应的模型预测值，回答用户问题，不要发散和联想内容。
如果没有结果时，回复“不知道”。
----------
预测结果：{query_result}
----------
用户问题：{query}
-----------
输出：
'''

SUMMARY_PROMPT_TPL = '''

请结合以下历史对话信息，和用户消息，总结出一个简洁、完整的用户消息。
用户当前消息为最高级，不要做过多的修改！！！！不要改变原意！，历史消息的总结尽量放在总体消息的后面！
直接给出总结好的消息，不需要其他信息。
如果和历史对话消息没有关联，直接输出用户原始消息。
注意，仅补充内容，不能改变原消息的语义，和句式。

例如：
-----------
历史对话：
Human:鼻炎是什么引起的？\nAI:鼻炎通常是由于感染引起。
用户消息：吃什么药好得快？
-----------
输出：得了鼻炎，吃什么药好得快？

-----------
历史对话：
{chat_history}
-----------
用户消息：{query}
-----------
输出：
'''



PARSE_TOOLS_PROMPT_TPL = '''
你有权限使用以下工具，请优先尝试使用graph_func这个工具回复用户问题，
如若无法给出答案，则根据工具描述和用户问题，判断应该使用哪个工具，直接输出工具名称即可。
-----------
{tools_description}
-----------
用户问题：{query}
-----------
输出：
'''
