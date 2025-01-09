'''
从特定目录读取文本文件（包括 CSV、PDF 和 TXT 文件），
将其分割成小块，然后将这些小块文本向量化并存储在一个持久化数据库中
'''

from langchain.embeddings import HuggingFaceBgeEmbeddings
import os
from glob import glob #遍历文件夹下的所有文件
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

# 定义模型参数
model_name = "/home/zwfeng4/.cache/modelscope/hub/AI-ModelScope/bge-large-zh"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def doc2vec():
    # 定义文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800, #块的大小
        chunk_overlap = 80 #重叠部分
    )
    # 读取并分割文件
    dir_path = os.path.join(os.path.dirname(__file__), './data/input/') #目录的路径
    documents = []
    for file_path in glob(dir_path + '*.*'):
        loader = None
        print(file_path)
        if '.csv' in file_path:
            loader = CSVLoader(file_path,encoding='utf-8')
        if '.pdf' in file_path:
            print(file_path)
            loader = PyMuPDFLoader(file_path)
        if '.txt' in file_path:
            loader = TextLoader(file_path,encoding='utf-8')
        if loader:
            documents += loader.load_and_split(text_splitter)

    print(len(documents))

    if documents:
        print("正在初始化向量数据库")
        vdb = Chroma(
            embedding_function = model,
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db4')
        )
        # 将文档分块并添加到向量数据库中
        chunk_size = 10
        for i in range(0, len(documents), chunk_size):
            texts = [doc.page_content for doc in documents[i:i+chunk_size]]
            metadatas = [doc.metadata for doc in documents[i:i+chunk_size]]
            vdb.add_texts(texts, metadatas)
            print(i)
        print("持久化向量数据库------")
        vdb.persist()


if __name__ == '__main__':
    doc2vec()