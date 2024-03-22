from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

################ Indexing ################
# Load
loader = PyMuPDFLoader("./1999_30problems.pdf")
PDF_data = loader.load()
# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=5, separators=["\nQ"])
all_splits = text_splitter.split_documents(PDF_data)  

# Embed and store (embedding model, you can search MTEB in HuggingFace)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# get embedding dimension
query_result = embedding_model.embed_query("embedding test")
print(len(query_result))

# save to disk
persist_directory = 'db-all-MiniLM-L6-v2' 
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding_model, persist_directory=persist_directory)

# Testing embedding_model with documents
result = vectordb.similarity_search("question you want to ask")
print(result[0].page_content)

################ loading LLM model ################
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from os.path import dirname

model_path = (dirname(__file__)) + "/model/"
llm = LlamaCpp(
    model_path=f"{model_path}\Breeze-7B-Instruct-v0.1-Q8_0.gguf",
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

question = "請問1999的服務範圍有哪些？"
print("=========== model_result ================")
llm.invoke(question)

################ define prompt template ################
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

RAG_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    {context}: 你是 1999 新北市政府的服務範圍查詢員。
    Instruction: 幫我回答以下問題：如果在文件中找不到相關信息，可以試著使用 google 搜尋功能，再找不到答案，請回答：“我無法回答這個問題”。
    Question: {question}
    Answer: 
    """
)


################ Augmentation: query + retrieved information ################
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": RAG_prompt},    
    verbose=True,
)

print("=========== RAG_result ================")
qa.invoke(question)
