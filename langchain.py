# from langchain_community.llms.tongyi import Tongyi

# model = Tongyi(model = "qwen-max")

# # res = model.invoke(input = "你是谁？")

# # print(res)

# res = model.stream(input = "你是谁？")

# for chunk in res:
#     print(chunk, end="", flush=True)



# from langchain_community.chat_models.tongyi import ChatTongyi
# # from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# model = ChatTongyi(model = "qwen3-max")

# # messages = [
# #     SystemMessage(content = "你是一个边塞诗人。"),
# #     HumanMessage(content = "写一首唐诗"),
# #     AIMessage(content = "边塞风光好，\n长城蜿蜒绕。\n烽火连三月，\n家书抵万金。"),
# #     HumanMessage(content = "按照上面的示例")
# # ]
# messages = [
#     ("system", "你是一个边塞诗人。"),
#     ("human", "写一首唐诗"),
#     ("ai", "边塞风光好，\n长城蜿蜒绕。\n烽火连三月，\n家书抵万金。"),
#     ("human", "按照上面的示例")
# ]

# res = model.stream(input = messages)

# for chunk in res:
#     print(chunk.content, end="", flush=True)





# from langchain_community.embeddings import DashScopeEmbeddings

# model = DashScopeEmbeddings()

# print(model.embed_query("我喜欢你。"))
# print(model.embed_documents(["我喜欢你。", "我讨厌你。"]))



# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms.tongyi import Tongyi

# prompt_template = PromptTemplate.from_template(
#     "我的名字是{名字}，我喜欢{爱好}。请为我推荐一些适合我的职业。"
#     )

# # prompt_text = prompt_template.format(名字="小明", 爱好="编程")
# model = Tongyi(model = "qwen-max")
# # res = model.invoke(input = prompt_text)
# # print(res)

# chain = prompt_template | model
# res = chain.invoke(input= {"名字": "小明", "爱好": "编程"})
# print(res)



from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate.from_template(
    "单词: {输入}\n反义词: {输出}"
)

examples = [
    {"输入": "高兴", "输出": "难过"},
    {"输入": "大", "输出": "小"},
    {"输入": "快", "输出": "慢"}
]



few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,    #示例数据模板
    examples=examples,          #示例数据列表，每个元素是一个字典，包含示例数据的输入和输出
    prefix="告知我单词的反义词，我给出一下例子",            #前缀文本，位于示例数据之前
    suffix="基于前面的例子，告诉我{输入}的反义词",            #后缀文本，位于示例数据之后
    input_variables=["输入"],   #输入变量列表
)

prompt_text = few_shot_prompt.invoke(input={"输入": "漂亮"}).to_string()
print(prompt_text)

from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model = "qwen-max")


res = model.invoke(input = prompt_text)
print(res)