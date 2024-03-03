from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
import os
def persist_text_information(document_path):
    OPENAI_API_KEY = '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'
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    loader = PyPDFLoader(document_path)
    pages = loader.load()
    print('pages:')
    print(pages)
    CHUNK_SIZE, CHUNK_OVERLAP = 5000, 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    text_splits = text_splitter.split_documents(pages)
    print("text_splits")
    print(text_splits)

    embeddings_text = OpenAIEmbeddings()
    MILVUS_HOST, MILVUS_PORT = 'localhost', 19530
    text_vectorstore = Milvus.from_documents(
        text_splits,
        embeddings_text,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    print("text_vectorstore")
    print(text_vectorstore.__dict__)

    print("Similarity")
    sim = text_vectorstore.as_retriever(search_kwargs={'k': 10})
    docs = sim.get_relevant_documents("Salaries and employee benefits")
    print(docs)

if __name__ == '__main__':
    persist_text_information('C:/Users/agarwal_ak/Desktop/reg-knowledge-bot/Uploads/Y9C_inst.pdf')