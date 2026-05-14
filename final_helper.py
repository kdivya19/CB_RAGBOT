# ═══════════════════════════════════════════════════════════════
# Final Helper — RAG with Token Tracking + Auto LLM Switching
#                + Conversation Memory
# ═══════════════════════════════════════════════════════════════

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

import os 

import logging
from dotenv import load_dotenv
from token_tracker import tracker

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── LLM Setup ──────────────────────────────────────────────

# PRIMARY: Gemini 2.5 Flash (free tier)
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# FALLBACK: Groq (free tier — Llama 3.3 70B)
# Get a free API key at https://console.groq.com
fallback_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# ─── Smart LLM Selector ─────────────────────────────────────

def get_active_llm():
    """Returns the right LLM based on today's token usage."""
    if tracker.is_limit_reached():
        logger.info("⚡ Token limit reached — switching to Groq (Llama 3.3 70B)")
        return fallback_llm
    return primary_llm


# ─── Embeddings & Vector DB ─────────────────────────────────

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


# ─── Session Memory Store ───────────────────────────────────

session_store = {}  # {session_id: ChatMessageHistory}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


# ─── Token Counting Helper ──────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English text)."""
    return max(1, len(text) // 4)


# ─── Build the QA Chain ─────────────────────────────────────

def get_qa_chain():
    vectordb = FAISS.load_local(
        vectordb_filepath,
        instruct_embedding,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    # ─── Prompt with chat history support ───
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for Codebasics.
Given the following context and conversation history, answer the user's question.
If the answer is not found in the context, say "I don't know the answer, please drop an email to info@codebasics.io"

CONTEXT:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ─── Custom chain function with token tracking + LLM switching ───
    def rag_with_tracking(inputs: dict) -> str:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # 1. Retrieve context
        docs = retriever.invoke(question)
        context = format_docs(docs)

        # 2. Pick the right LLM
        llm = get_active_llm()
        active_name = tracker.get_active_llm_name()
        logger.info(f"🤖 Using LLM: {active_name} | {llm.__class__.__name__}")

        # 3. Format the prompt
        formatted = prompt.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        })

        # 4. Call the LLM
        response = llm.invoke(formatted)

        # 5. Extract text (ChatGroq returns AIMessage, Gemini returns str)
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        # 6. Track token usage
        input_tokens = estimate_tokens(str(formatted))
        output_tokens = estimate_tokens(answer)
        stats = tracker.add_tokens(input_tokens, output_tokens)
        logger.info(
            f"📊 Tokens today: {stats['total_tokens']:,} / {stats['daily_limit']:,} "
            f"({stats['usage_percent']}%) | Requests: {stats['request_count']}"
        )

        return answer

    # Wrap as a LangChain Runnable
    chain = (
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | RunnableLambda(rag_with_tracking)
    )

    # ─── Wrap with conversation memory ───
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


if __name__ == "__main__":
    # ⚠️ Step 1: First time only run this
    # vectordb_creation()

    chain = get_qa_chain()

    # Test with a session
    config = {"configurable": {"session_id": "test-session"}}
    print(chain.invoke({"question": "Do you have java course?"}, config=config))
    print("\n--- Stats ---")
    print(tracker.get_stats())