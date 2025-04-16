import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# === PIPELINE STAGE BASE CLASS ===
class PipelineStage:
    def __init__(self):
        self.next_stage = None

    def set_next(self, stage):
        self.next_stage = stage
        return stage

    def execute(self, data):
        raise NotImplementedError("Each stage must implement execute() method.")

# === STAGE 1: LOAD PDF TEXT ===
class LoadPDFStage(PipelineStage):
    def execute(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return self.next_stage.execute(text) if self.next_stage else text

# === STAGE 2: SPLIT TEXT INTO CHUNKS ===
class SplitTextStage(PipelineStage):
    def execute(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
        chunks = splitter.split_text(text)
        return self.next_stage.execute(chunks) if self.next_stage else chunks

# === STAGE 3: CREATE AND SAVE VECTOR STORE ===
class VectorStoreStage(PipelineStage):
    def execute(self, chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return self.next_stage.execute("faiss_index") if self.next_stage else "faiss_index"

# === STAGE 4: QUERY STAGE USING GEMINI MODEL ===
class QueryStage(PipelineStage):
    def __init__(self, model_name, question):
        super().__init__()
        self.model_name = model_name
        self.question = question

    def execute(self, vector_store_path):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(self.question)

        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the context, just say "Answer is not available in the context."

        Context:
        {context}
        Question: 
        {question}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        response = chain({"input_documents": docs, "question": self.question}, return_only_outputs=True)
        answer = response["output_text"]

        # Save to chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"question": self.question, "answer": answer})

        return answer

# === FUNCTION TO BUILD PIPELINE ===
def build_pipeline(question=None):
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    model = "models/gemini-1.5-pro" if "models/gemini-1.5-pro" in available_models else available_models[0]

    loader = LoadPDFStage()
    splitter = loader.set_next(SplitTextStage())
    vector_creator = splitter.set_next(VectorStoreStage())

    if question:
        query_stage = vector_creator.set_next(QueryStage(model, question))
    return loader

# === SIDEBAR ===
def display_sidebar():
    with st.sidebar:
        st.image("https://i.imgur.com/ZyXkVwP.png", caption="PDF Chatbot ")
        st.title("PDF Upload")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        process_button = st.button(" Submit & Process")
        return pdf_docs, process_button

# === CHAT DISPLAY ===
def display_chat():
    st.subheader("🗨️ Chat History")
    if "chat_history" in st.session_state:
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**Q: {chat['question']}**\n\n💬 {chat['answer']}")
    else:
        st.info("Upload PDFs and ask a question to begin.")

# === MAIN APP ===
def main():
    st.set_page_config("PDF Chatbot (Pipeline)", page_icon="📄")
    st.header(" PDF Chatbot (Pipeline Architecture)")

    pdf_docs, process_button = display_sidebar()
    user_question = st.text_input("Ask a question from the PDFs...")

    if process_button and pdf_docs:
        with st.spinner("Processing PDFs..."):
            pipeline = build_pipeline()
            pipeline.execute(pdf_docs)
            st.success(" PDFs processed successfully!")

    if user_question:
        with st.spinner("Generating answer..."):
            pipeline = build_pipeline(question=user_question)
            answer = pipeline.execute("faiss_index")
            st.write(" **Answer:**", answer)

    display_chat()

    st.markdown(
        "<hr><center><small>Built with  using Pipeline Architecture by Induja</small></center>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
