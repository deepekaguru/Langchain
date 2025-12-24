import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- Paths / folders ----------

POLICY_DIR = "data/policies"
VECTOR_DIR_POLICIES = "vectorstore/policies"

CSV_PATH = "data/credit/creditcard.csv"
VECTOR_DIR_CASES = "vectorstore/fraud_cases"

app = FastAPI(
    title="Fraud Investigation Copilot API",
    description="Explain and justify fraud flags using policy-based RAG + similar historical cases.",
    version="0.2.0",
)

qa_chain = None              # policy RAG chain
cases_retriever = None       # retriever over historical fraud cases


# ---------- Vector store builders ----------


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
        persist_directory=VECTOR_DIR_POLICIES,
    )
    vectordb.persist()
    return vectordb


def build_cases_vectorstore():
    """
    Load creditcard.csv, keep fraud rows (Class == 1),
    turn them into short text summaries, and store in Chroma.
    """
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(
            f"CSV file not found at {CSV_PATH}. "
            "Place creditcard.csv there or update CSV_PATH."
        )

    df = pd.read_csv(CSV_PATH)

    if "Class" not in df.columns:
        raise RuntimeError("Expected 'Class' column in CSV (fraud label).")

    fraud_df = df[df["Class"] == 1].copy()

    # To keep it light, optionally sample if too many rows
    if len(fraud_df) > 2000:
        fraud_df = fraud_df.sample(n=2000, random_state=42)

    summaries = []
    for idx, row in fraud_df.iterrows():
        amount = row.get("Amount", None)
        time_val = row.get("Time", None)

        summary = (
            f"Confirmed fraud case: amount {amount} units, "
            f"activity time index {time_val}. "
            "Pattern is consistent with suspicious velocity and anomaly behavior."
        )
        summaries.append(summary)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_texts(
        texts=summaries,
        embedding=embeddings,
        persist_directory=VECTOR_DIR_CASES,
    )
    vectordb.persist()
    return vectordb


# ---------- Chains / retrievers ----------


def make_policy_qa_chain():
    """Create a RAG chain that uses policy docs to answer questions."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DIR_POLICIES,
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

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def make_cases_retriever():
    """Create a retriever over historical fraud cases."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DIR_CASES,
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})


# ---------- Core logic ----------


def explain_transaction(tx: dict, chain) -> str:
    """Given a transaction dict, ask the policy RAG chain whether to flag it."""
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

    answer = chain.invoke(question)
    return answer


def find_similar_cases(tx: dict, retriever, k: int = 3):
    """
    Use the fraud cases retriever to find similar historical fraud examples.
    Returns a list of short text summaries.
    """
    desc = (
        f"Transaction with amount {tx['amount']} {tx.get('currency', 'USD')} "
        f"and recent activity count {tx['txn_velocity_24h']} in the last 24 hours."
    )

    # In new LangChain, retrievers are runnables: use .invoke() instead of get_relevant_documents()
    docs = retriever.invoke(desc)
    return [d.page_content for d in docs[:k]]



# ---------- Request models ----------


class Transaction(BaseModel):
    amount: float
    currency: str = "USD"
    country: str
    account_age_days: int
    txn_velocity_24h: int
    device_change: bool


# ---------- FastAPI lifecycle ----------


@app.on_event("startup")
def startup_event():
    """Initialize API: load env, build vectorstores, create RAG chain & retriever."""
    global qa_chain, cases_retriever

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    # Policies vector store
    if not os.path.exists(VECTOR_DIR_POLICIES):
        os.makedirs(VECTOR_DIR_POLICIES, exist_ok=True)
        print("Building policy vector store on startup...")
        build_policy_vectorstore()
    else:
        print("Using existing policy vector store.")

    qa_chain = make_policy_qa_chain()
    print("RAG policy chain initialized.")

    # Fraud cases vector store
    if not os.path.exists(VECTOR_DIR_CASES):
        os.makedirs(VECTOR_DIR_CASES, exist_ok=True)
        print("Building fraud cases vector store on startup...")
        build_cases_vectorstore()
    else:
        print("Using existing fraud cases vector store.")

    cases_retriever = make_cases_retriever()
    print("Fraud cases retriever initialized.")


# ---------- Endpoints ----------


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Fraud Investigation Copilot API is running.",
    }


@app.post("/explain")
def explain(tx: Transaction):
    """
    Explain whether a transaction should be flagged, using:
    - Policy-based RAG (rules)
    - Similar historical fraud cases from creditcard.csv
    """
    global qa_chain, cases_retriever
    if qa_chain is None:
        raise RuntimeError("QA chain not initialized")

    explanation = explain_transaction(tx.dict(), qa_chain)

    # Extract rule IDs mentioned in the explanation, e.g. "AML-TRX-001"
    rule_ids = sorted(set(re.findall(r"AML-TRX-\d{3}", explanation)))

    # Simple heuristic decision from explanation text
    lower = explanation.lower()
    if "should be flagged" in lower or "flag this transaction" in lower:
        decision = "FLAG"
    elif "not explicitly covered" in lower:
        decision = "REVIEW"
    else:
        decision = "REVIEW"

    similar_cases = []
    if cases_retriever is not None:
        similar_cases = find_similar_cases(tx.dict(), cases_retriever, k=3)

    return {
        "decision": decision,
        "rules_triggered": rule_ids,
        "similar_cases": similar_cases,
        "explanation": explanation,
        "input": tx,
    }
