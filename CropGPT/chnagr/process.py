from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.prompts import PromptTemplate
from utils import get_embeddings_model  # 使用绝对导入
vdb = Chroma(
        embedding_function=get_embeddings_model(),
        persist_directory=os.path.join(os.path.dirname(__file__), "./tempDB/vdb")
)

# 定义文档分割标准
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 50
)

# 拿到目标文件夹路径，并分割其中文档
def loadFile():
    documents = [] # 装文档内容用
    loader = TextLoader("content.txt")
    print(loader)
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error during loading: {e}")

    documents = loader.load_and_split(text_spliter)
    return documents

# 将分割完成后的文档进行向量化
def embeddingText():
    try:
        documents = loadFile()
        chunk_size = 10
        for i in range(0, len(documents), chunk_size):
            texts = [doc.page_content for doc in documents[i:i + chunk_size]]
            metadatas = [doc.metadata for doc in documents[i:i + chunk_size]]
            vdb.add_texts(texts, metadatas)
            print(i)
        vdb.persist()
        print("完成向量化")
    except Exception as e:
        print(f"向量化过程中出错: {e}")