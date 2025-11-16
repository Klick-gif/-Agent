from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
# from create_llm import create_local_qwen_model
# 创建LLM
llm = ChatOpenAI(
    # model="qwen3-235b-a22b-instruct-2507",
    model = "qwen-plus",
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key="sk-eb8efc4f031c464c8db80722d7c7dbd1",
    streaming=True
)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key="sk-eb8efc4f031c464c8db80722d7c7dbd1",
)

# 创建本地Qwen模型，需要在本地显存高的情况下使用
# local_llm = create_local_qwen_model()

