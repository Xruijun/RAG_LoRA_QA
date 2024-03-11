# 导入应用程序所需的库和模块。
import os

import torch.cuda
from langchain_community.llms.chatglm3 import ChatGLM3  # 导入ChatGLM3模型，用于生成语言模型响应。
import gradio as gr  # 导入Gradio库，用于创建Web界面。
from langchain.document_loaders import DirectoryLoader  # 用于从目录加载文档。
from langchain.prompts import PromptTemplate  # 用于创建提示模板。
from langchain.text_splitter import CharacterTextSplitter  # 用于将文本文档分割成更小的块。
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # 用于使用HuggingFace模型生成嵌入。
from langchain.vectorstores import Chroma  # 用于存储和检索文档的向量化表示。
from langchain.chains import RetrievalQA  # 用于构建基于检索的问答系统。
from langchain.schema.messages import AIMessage  # 用于创建可用作提示或响应的AI消息。


def load_documents(directory='documents'):
    loader = DirectoryLoader(directory)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_splitter.split_documents(document)
    return split_docs


# 加载embedding模型
def load_embedding_model():
    encode_kwargs = {'normalize_embeddings': False}
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    return HuggingFaceEmbeddings(
        model_name='text2vec-base-chinese',
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder='text2vec-base-chinese'  # cache_folder是用于加载本地模型，后面接目录名
    )


def store_chroma(documents, embeddings):
    db = Chroma.from_documents(documents, embeddings, persist_directory="VectorStore")
    db.persist()
    return db


embeddings = load_embedding_model()

if not os.path.exists("VectorStore"):
    documents = load_documents()
    db = store_chroma(documents, embeddings)
else:
    db = Chroma(persist_directory="VectorStore", embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents(query="支原体")

endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

messages = [
    AIMessage(content="欢迎询问我问题")
]

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    prefix_messages=messages,
    top_p=0.9,
)

prompt = PromptTemplate.from_template(template="""根据给的context，给出对应的回答
context:{context}
question:{question}
""")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
    chain_type_kwargs={"prompt": prompt}
)


def chat(question, history):
    response = qa.run(question)
    return response


demo = gr.ChatInterface(chat)
demo.launch()
