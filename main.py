import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Folders
POLICY_DIR = "data/policies"
VECTOR_DIR = "vectorstore/policies"


def build_policy_vectorstore():
    """Read policy .txt files, split them, embed, and save to Chroma."""
    if not os.path.isdir(POLICY_DIR):
        raise RuntimeError(f"Policy directory not found: {POLICY_DIR}")

    texts = []
    for fname in os.listdir(POLICY_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(POLICY_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    if not texts:
        raise RuntimeError(f"No .txt files found in {POLICY_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    docs = splitter.create_documents(texts)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
    )
    # persist() is technically not needed on new Chroma, but harmless
    vectordb.persist()
    return vectordb


def make_policy_qa_chain():
    """Create a RAG chain that uses policy docs to answer questions."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DIR,
    )

    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template(
        """
You are a banking fraud analyst assistant.

Use ONLY the provided policy documents to answer.
If the question is not covered by policy, say:
"This is not explicitly covered in our rules."

Policy context:
{context}

Question:
{question}
"""
    )

    # New-style RAG chain using runnables
    chain = (
        {
            "context": retriever,              # retrieval step
            "question": RunnablePassthrough()  # pass the question through
        }
        | prompt               # fill prompt with context + question
        | llm                  # call the LLM
        | StrOutputParser()    # return a plain string
    )

    return chain


def explain_transaction(tx: dict, qa_chain) -> str:
    """Given a transaction dict, ask the RAG policy chain whether to flag it."""
    desc = (
        f"Amount: {tx['amount']} {tx.get('currency', 'USD')}, "
        f"Country: {tx['country']}, "
        f"Account age: {tx['account_age_days']} days, "
        f"Transactions in last 24h: {tx['txn_velocity_24h']}, "
        f"Device change: {tx['device_change']}."
    )

    question = (
        "Given our fraud rules, should this transaction be flagged? "
        "Explain clearly and reference rule IDs if relevant.\n\n"
        f"Transaction details: {desc}"
    )

    answer = qa_chain.invoke(question)
    return answer


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    # 1) Sanity check – basic hello from LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    greeting = llm.invoke(
        "You are a banking fraud assistant. Say hello in one sentence."
    )
    print("LLM response:")
    print(greeting.content)
    print("-" * 60)

    # 2) Build vector store from fraud policy (only first time)
    if not os.path.exists(VECTOR_DIR):
        print("Building policy vector store...")
        os.makedirs(VECTOR_DIR, exist_ok=True)
        build_policy_vectorstore()
    else:
        print("Policy vector store already exists, skipping build.")

    # 3) Create the RAG chain
    qa_chain = make_policy_qa_chain()

    # 3a) Simple policy question
    question = (
        "According to our rules, when should we flag a high-value "
        "cross-border transaction?"
    )
    print("\nRAG answer to policy question:")
    print(qa_chain.invoke(question))
    print("-" * 60)

    # 3b) Explain a concrete transaction
    sample_tx = {
        "amount": 5200,
        "currency": "USD",
        "country": "MX",
        "account_age_days": 12,
        "txn_velocity_24h": 6,
        "device_change": True,
    }

    print("Explaining sample transaction:")
    tx_explanation = explain_transaction(sample_tx, qa_chain)
    print(tx_explanation)


if __name__ == "__main__":
    main()
