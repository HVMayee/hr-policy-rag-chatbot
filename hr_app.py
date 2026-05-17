import streamlit as st
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Page setup
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide"
)

st.title("👩‍💼 HR Policy Assistant")
st.write("Ask questions from HR policy data.")

# Load vector DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # If vector database does not exist, create it from cleaned_hr_policy.csv
    if not os.path.exists("hr_vector_db"):
        df = pd.read_csv("cleaned_hr_policy.csv")

        documents = []

        for _, row in df.iterrows():
            doc = Document(
                page_content=row["search_text"],
                metadata={
                    "policy_id": str(row["policy_id"]),
                    "policy_category": str(row["policy_category"]),
                    "policy_title": str(row["policy_title"])
                }
            )
            documents.append(doc)

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="hr_vector_db"
        )
    else:
        vector_db = Chroma(
            persist_directory="hr_vector_db",
            embedding_function=embeddings
        )

    return vector_db

vector_db = load_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


def generate_hr_answer(question, docs):
    answer = "Based on the retrieved HR policy records, here is the answer:\n\n"

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        answer += f"{i}. Policy ID: {metadata.get('policy_id')}\n"
        answer += f"   Category: {metadata.get('policy_category')}\n"
        answer += f"   Policy Title: {metadata.get('policy_title')}\n"

        policy_lines = doc.page_content.split("\n")
        policy_text = ""

        for line in policy_lines:
            if line.startswith("Policy Details:"):
                policy_text = line.replace("Policy Details:", "").strip()

        answer += f"   Policy Details: {policy_text}\n\n"

    answer += "Summary:\n"

    question_lower = question.lower()

    if "carry forward" in question_lower or "unused leave" in question_lower or "unused leaves" in question_lower:
        answer += "Employees can carry forward up to 5 unused annual leaves to the next calendar year.\n\n"
    elif "annual leave" in question_lower or "paid leave" in question_lower:
        answer += "Employees are eligible for 18 paid leaves per calendar year.\n\n"
    elif "sick leave" in question_lower:
        answer += "Employees are eligible for 7 sick leaves per calendar year, and sick leave should be informed to the manager as soon as possible.\n\n"
    elif "work from home" in question_lower or "wfh" in question_lower:
        answer += "Employees may work from home up to 2 days per week with prior approval from their reporting manager.\n\n"
    elif "travel reimbursement" in question_lower or "travel claim" in question_lower:
        answer += "Employees must submit travel reimbursement claims with valid bills within 7 days of trip completion.\n\n"
    elif "meal expense" in question_lower or "food expense" in question_lower:
        answer += "Meal expenses during business travel can be claimed up to the approved daily limit with valid receipts.\n\n"
    elif "notice period" in question_lower or "resignation" in question_lower:
        answer += "Employees must serve a notice period of 60 days after resignation unless otherwise approved by HR.\n\n"
    elif "medical reimbursement" in question_lower:
        answer += "Medical reimbursement claims must be submitted with valid medical bills and prescriptions.\n\n"
    elif "confidential" in question_lower or "company information" in question_lower:
        answer += "Employees must not share confidential company or customer information with unauthorized persons.\n\n"
    else:
        answer += "Please refer to the retrieved HR policy records above for the relevant policy details.\n\n"

    answer += "Suggested next action:\n"
    answer += "If the employee needs confirmation for a specific case, they should contact HR or their reporting manager."

    return answer


# User input
question = st.text_input("Ask your HR policy question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        docs = retriever.invoke(question)
        answer = generate_hr_answer(question, docs)

        st.subheader("HR Chatbot Answer")
        st.write(answer)

        st.subheader("Source Records Used")

        for i, doc in enumerate(docs, start=1):
            with st.expander(f"Source {i}"):
                st.write(doc.page_content)
                st.json(doc.metadata)