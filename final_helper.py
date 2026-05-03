# # helper_code.py
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_community.document_loaders import CSVLoader 
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate 
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import os 
# from dotenv import load_dotenv

# load_dotenv()

# # Global Init (FastAPI starts faster if these are ready)
# llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.3)
# instruct_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb_filepath = "faiss_index" 

# def vectordb_creation():
#     """Only run this ONCE locally to build the DB."""
#     loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
#     data = loader.load()
#     vectordb = FAISS.from_documents(documents=data, embedding=instruct_embedding)
#     vectordb.save_local(vectordb_filepath)

# def get_qa_chain():
#     vectordb = FAISS.load_local(
#         vectordb_filepath,
#         instruct_embedding,
#         allow_dangerous_deserialization=True
#     )
    
#     # FIX: Correct way to set threshold
#     retriever = vectordb.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={"score_threshold": 0.7, "k": 3} 
#     )

#     prompt_template = """
#     Given the following context and a question, generate an answer based on this context only.
#     If the answer is not found in the context, say "I don't know the answer, please drop an email to [info@codebasics.io](mailto:info@codebasics.io)"

#     CONTEXT:
#     {context}

#     QUESTION:
#     {question}
#     """

#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return chain

# if __name__=="__main__":
#     # vectordb_creation()  <-- Comment this out! Don't rebuild every time.
#     chain = get_qa_chain()
#     # FIX: Use .invoke() for LCEL chains
#     print(chain.invoke("Do you have java course?"))






from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader 
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
 
import os 
from dotenv import load_dotenv

load_dotenv()

# Global Init (FastAPI starts faster if these are ready)
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)


instruct_embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb_filepath = "faiss_index" 

def vectordb_creation():
    """Run this ONCE locally to build the DB."""
    loader = CSVLoader(
        file_path='codebasics_faqs.csv',
        source_column='prompt'
    )

    data = loader.load()

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=instruct_embedding
    )

    vectordb.save_local(vectordb_filepath)


def get_qa_chain():
    vectordb = FAISS.load_local(
        vectordb_filepath,
        instruct_embedding,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 4} 
    )

    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    If the answer is not found in the context, say "I don't know the answer, please drop an email to info@codebasics.io"

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    # ⚠️ Step 1: First time only run this
    vectordb_creation()

    chain = get_qa_chain()
    print(chain.invoke("Do you have java course?"))