# RAG_example - using rag with local model in langchain

RAG 運作流程:

![image](https://github.com/hzishan/RAG_example/assets/104684284/3ed59ae1-6d08-4aab-978f-7aed4f0040a9)

圖片來源: https://reurl.cc/09NE0Y

RAG 的 Framework : 主要分為 Retriever、Augmentation、Generator
- Retriver: 通過將 query 與 index 向量進行比較，獲取相關文檔，也稱為“相關文檔”。
- Augmentation: 將相關文檔與原始提示結合作為額外的上下文。 
- Generator: 將組合的文本和提示傳遞給模型進行響應生成，然後準備為系統向用戶的最終輸出



## 程式執行:

把需要使用的 local model (.gguf) 放在同層目錄 model 資料夾中
### Indexing 
將想要嵌入的文件，做前處理: embed + store
- embed: 可以去 [huggingface MTEB](https://huggingface.co/spaces/mteb/leaderboard) 找最適合的 embedding model
- store: 目前使用 vectorStore 的方式，想是其他方法可以參考[langchain retivers](https://python.langchain.com/docs/modules/data_connection/retrievers/)的用法

![image](https://github.com/hzishan/RAG_example/assets/104684284/28fee892-b7bc-4aaa-935f-c13f72b0c852)
圖片來源: https://python.langchain.com/docs/use_cases/question_answering/

similarity_search 可以查看 embedding 出來的資料相關性。
```
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

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

# load from disk
# docs = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
# Testing embedding_model with documents

result = vectordb.similarity_search("question you want to ask")
print(result[0].page_content)
```

### loading your local LLM
目前已經改成用 `llm.invoke()` 代替 `llm.run()`
可以先 output 出來，查看 rag 前後，模型回答的能力及相關性。

```
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
```
### Prompting
Prompt 的部分還沒搞懂，目前只是根據 [prompt engineering](https://www.promptingguide.ai/) 及 [這篇文章](https://github.com/EgoAlpha/prompt-in-context-learning/blob/main/PromptEngineering.md) 做為參考
```
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
```
### Combine Prompt and Generate answer
另一種 [LLMChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html#langchain.chains.llm.LLMChain) 的方式也可以把 prompt 跟 LLM 串起來，實作可以參考上面的[同篇文章](https://reurl.cc/09NE0Y)
```
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": RAG_prompt},    
    verbose=True,
)

print("=========== RAG_result ================")
qa.invoke(question)
```
