from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

#load environmental variabless!

load_dotenv()


# load a pdf in my case its my calculus chapter pdf hehe
loader = PyPDFLoader("kemh112.pdf")
pages = loader.load()

# split into chunks , also overlap so the AI have yk persisted memory
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)
if os.path.exists('./chromadb'):
    vectorstore = Chroma(
        persist_directory="./chromadb",
        embedding_function = embeddings
    )
    print("loaded from existing vectorstore")

else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory='./chromadb'
    )
    print("created a new vectorstore")

  
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model = 'openai/gpt-oss-20b'
)    

prompt = ChatPromptTemplate.from_template("""answer the question based only on the following context:{context}question: {question}""")

while True:
    question = input("Enter your question: ")
    if question.lower() == "quit":
        print("goodbyee!!")
        break
    docs = vectorstore.similarity_search(question,k=8)
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = prompt|llm|StrOutputParser()
    answer = chain.invoke({"context":context,"question":question})
    print(f"Answer: {answer}")
