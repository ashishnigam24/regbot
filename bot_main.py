from document_data_read.document_knowledge_bot import Document_Knowledge_Bot
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from utils.connection import get_milvus_details, get_document_llm_model_details
from SQL_data_read.database import LangchainStorage
from SQL_data_read.sql_knowledge_bot import SQL_knowledge_bot
from prettytable import PrettyTable
from langchain.chat_models import ChatOpenAI
import os
import time
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory


def process_query(document_knowledge_bot,sql_knowledge_bot,langchainstorage,vectorstore, memory):
    x = PrettyTable()
    x.field_names = ["Persona        ", "Conversation"]
    x.align = "l"
    x.column_width = 100
    x.max_table_width = 180
    global dbqa
    DOCUMENT_LLM_MODEL, DOCUMENT_LLM_MODEL_TEMPERATURE, DOCUMENT_LLM_MODEL_MAX_TOKENS, DOCUMENT_LLM_MODEL_TOP_P = get_document_llm_model_details()
    llm = ChatOpenAI(temperature=DOCUMENT_LLM_MODEL_TEMPERATURE,
                     model_name=DOCUMENT_LLM_MODEL,
                     max_tokens=DOCUMENT_LLM_MODEL_MAX_TOKENS,
                     top_p=DOCUMENT_LLM_MODEL_TOP_P,
                     openai_api_key=os.getenv("OPENAI_API_KEY")
                     )

    qa_prompt_text = document_knowledge_bot.set_qa_prompt_text()
    dbqa_text = document_knowledge_bot.build_retrieval_qa_text(vectorstore, llm, qa_prompt_text, memory)
    qa_prompt_table = document_knowledge_bot.set_qa_prompt_table()
    dbqa_table = document_knowledge_bot.build_retrieval_qa_table(vectorstore, llm, qa_prompt_table)
    while True:
        answered_flag = False
        question = input("Enter your Question or type 'q' to quit: ")
        if question == "q":
            break
        start_time = time.time()
        x.add_row(["", ""])
        x.add_row(["", ""])
        x.add_row(["You: ", question])
        x.add_row(["", ""])
        table_result = dbqa_table({'query': question})["result"]
        memory.save_context({"question": question}, {"output": table_result})
        text_result = dbqa_text({'query': question})
        memory.save_context({"question": question}, {"output": text_result["result"]})
        # print("memory:")
        # print(memory.load_memory_variables({}))
        reference_doc_set = set()
        for doc in text_result['source_documents']:
            reference_doc_set.add("file:///" + doc.metadata['source'].replace('\\',"/") + "#page=" + str(doc.metadata['page']))
        if 'No information found'.lower() not in table_result.lower():
            answered_flag = True
            x.add_row(["Ai Assistant (Table): ", table_result])
            x.add_row(["", ""])
            x.add_row(["", ""])
        if 'No information found'.lower() not in text_result["result"].lower():
            answered_flag = True
            x.add_row(["Ai Assistant (Text): ", text_result["result"]])
            for reference in reference_doc_set:
                if "file:///Table" not in reference:
                    x.add_row(["Reference : ", str(reference)])
            x.add_row(["", ""])
            x.add_row(["", ""])
        sql_response_obj, sql_query = sql_knowledge_bot.process_query(question)
        if sql_response_obj is None:
            struct_result = "Table doesn't exists or question is not valid"
        else:
            struct_result = langchainstorage.display_result(sql_response_obj)
            x.add_row(["Ai Assistant (Query): ", sql_query])
        if struct_result != None:
            answered_flag = True
            x.add_row(["Ai Assistant (Structured): ", struct_result])
            x.add_row(["", ""])
            x.add_row(["", ""])
        if not answered_flag:
            x.add_row(["Ai Assistant : ", "Sorry, I don't have enough information for your question"])
            x.add_row(["", ""])
            x.add_row(["", ""])
        print(x)
        end_time = time.time()
        print(f"Answer generated in {end_time - start_time} seconds.")


if __name__ == '__main__':
    document_knowledge_bot = Document_Knowledge_Bot()
    sql_knowledge_bot = SQL_knowledge_bot()
    embeddings = OpenAIEmbeddings()
    langchainstorage = LangchainStorage()
    MILVUS_HOST, MILVUS_PORT = get_milvus_details()
    vectorstore = Milvus.from_documents(
        '',
        embeddings,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    #memory = ConversationBufferWindowMemory(memory_key="history", input_key="question", k=4)
    memory = ConversationBufferMemory()
    process_query(document_knowledge_bot,sql_knowledge_bot,langchainstorage,vectorstore, memory)
