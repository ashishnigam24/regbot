from document_data_read.document_knowledge_base import Document_Knowledge_Base
from utils.connection import get_upload_directory, set_dotenv
import os

if __name__ == "__main__":
    set_dotenv()
    UPLOADS_DIRECTORY = get_upload_directory()
    os.makedirs(UPLOADS_DIRECTORY, exist_ok=True)
    document_knowledge_base = Document_Knowledge_Base()
    table_vectorstore = None
    text_vectorstore = None
    choice_doc_type = None
    choice_dict = {1: 'text',
                   2: 'table',
                   3: 'both',
                   4: 'structured data'}
    while True:
        print("Menu:")
        print("1. Upload Document")
        print("2. Delete Documents")
        print("3. Quit")

        choice = input("Enter your choice: ")
        if choice == '1':
            print("Menu:")
            print("1. Document contains only text")
            print("2. Document contains only table")
            print("3. Document contains both text and table")
            choice_doc_type = int(input("Enter your choice: "))
            table_vectorstore, text_vectorstore = document_knowledge_base.upload_document(choice_dict[choice_doc_type])
        elif choice == '2':
            document_knowledge_base.delete_documents()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please select a valid option.")
#     except(Exception) as e:
#        print(e)

# Documents to read:
# y9c050331cci.pdf
# lcr230331.pdf
# FR_Y-9C20231231_i.pdf
