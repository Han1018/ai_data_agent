import os
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, VectorSearchVectorStore
from langchain.chains import RetrievalQA
from vertexai.preview.generative_models import GenerativeModel
from langchain.tools import Tool
import re
import json
from langchain_core.documents import Document
from config import (PROJECT_ID, REGION, BUCKET, INDEX_ID, 
                    ENDPOINT_ID, BUCKET_URI, 
                    MODEL_NAME, EMBEDDING_MODEL_NAME
                    )
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain # .retrieval.create_retrieval_chain
from prompt import RAG_SPLIT_QUERY_PROMPT, RAG_REPORT_PROMPT, RAG_CHAT_PROMPT

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)


my_index = aiplatform.MatchingEngineIndex(INDEX_ID)
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT_ID)

embedding_model = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# ✅ 建立向量資料庫
vector_store = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=REGION,
    gcs_bucket_name=BUCKET,
    index_id=my_index.name,
    endpoint_id=my_index_endpoint.name,
    embedding=embedding_model,
)
retriever = vector_store.as_retriever()

def extract_info_from_query(llm, query: str):
    
    """使用 Gemini 1.5 Pro 解析 query，提取 Company Name、CALENDAR_YEAR 和 CALENDAR_QTR"""

    prompt = RAG_SPLIT_QUERY_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    extracted_text = response.strip()

    # 正則表達式解析
    company_match = re.search(r"Company Name:\s*(.+)", extracted_text)
    year_match = re.search(r"CALENDAR_YEAR:\s*(\d{4})", extracted_text)
    qtr_match = re.search(r"CALENDAR_QTR:\s*(Q[1-4])", extracted_text)

    company_name = company_match.group(1).strip() if company_match else None
    calendar_year = year_match.group(1).strip() if year_match else None
    calendar_qtr = qtr_match.group(1).strip() if qtr_match else None

    return company_name, calendar_year, calendar_qtr

# def query_rag_tool(query: str, IS_FISCAL=True, ROLE='cn', MODE='report'): # 仍要新增ROLE的filter功能
def query_rag_tool(query: str): # 仍要新增ROLE的filter功能
    """使用 Vertex AI Vector Search 進行檢索，並使用 Gemini 1.5 Pro 解析 Query"""
    llm = VertexAI(model_name=MODEL_NAME)
    parsed_data = json.loads(query)

    query = parsed_data["query"]
    IS_FISCAL = parsed_data["IS_FISCAL"]
    ROLE = parsed_data["ROLE"]
    MODE = parsed_data["MODE"]

    # 可訪問的公司列表
    cn_companies = ['Baidu', 'Tencent']
    kr_companies = ['Samsung']

    # MODE
    if MODE == 'report':
        SYSTEM_PROMPT = RAG_REPORT_PROMPT # 改：真正的PROMPT
        K=1000
        ROLE = 'gb'       # 不分權限
        IS_FISCAL = False # 歷年
    else:
        SYSTEM_PROMPT = RAG_CHAT_PROMPT   # 改：真正的PROMPT
        K=10

    # 🔍 使用 Gemini 解析 Query，提取資訊
    company_name, year, qtr = extract_info_from_query(llm, query)

    print(f"Extracted Info:\n - Company Name: {company_name}\n - YEAR: {year}\n - QTR: {qtr}")

    rag_res = {
        "answer": "",
        "sources": None,
        "metadata": None,
        "source_documents": None, 
        "extracted_info": {
            "Company Name": company_name,
            "IS_FISCAL": IS_FISCAL,
            "YEAR": year,
            "QTR": qtr,
            "MODE": MODE
        }
    }

    # 📌 設定權限過濾
    if ROLE.upper() == 'CN' and company_name not in cn_companies:
        print(f"🔴 Access Denied: '{company_name}' is not available for CN role.")
        rag_res["answer"] = "Access Denied: You are only allowed to view CN companies."
        return rag_res

    if ROLE.upper() == 'KR' and company_name not in kr_companies:
        print(f"🔴 Access Denied: '{company_name}' is not available for KR role.")
        rag_res["answer"] = "Access Denied: You are only allowed to view KR companies."
        return rag_res
    
    # 📌 設定檢索條件（如果有）
    filters = []  # ✅ 用來存儲所有 `Namespace` 篩選條件
    numeric_filters = []  # ✅ 用來存儲所有 `NumericNamespace` 篩選條件

    if company_name:
        filters.append(Namespace(name="Company Name", allow_tokens=[company_name]))  # ✅ 正確加入 filters
    
    if IS_FISCAL: # 財年
        if year:
            numeric_filters.append(NumericNamespace(name="FISCAL_YEAR", value_float=float(year), op="EQUAL"))  # ✅ 加入數值篩選     # 財年
        if qtr:
            filters.append(Namespace(name="FISCAL_QTR", allow_tokens=[qtr]))  # ✅ 加入 filters     # 財年
    else:
        if year:
            numeric_filters.append(NumericNamespace(name="CALENDAR_YEAR", value_float=float(year), op="EQUAL"))  # ✅ 加入數值篩選 # 歷年
        if qtr:
            filters.append(Namespace(name="CALENDAR_QTR", allow_tokens=[qtr]))  # ✅ 加入 filters # 歷年

    print(filters, numeric_filters)
    retriever.search_kwargs = {
    "k": 10,
    "filter": filters if filters else None,  # ✅ 確保 `filter` 只出現一次
    "numeric_filter": numeric_filters if numeric_filters else None,  # ✅ 正確加入數值篩選
    }

    prompt = ChatPromptTemplate([
        # ('system', f'Answer the user\'s questions in zh-tw, i.e. traditional Chinese, based on the context provided below:\n\n{{context}}\n If you don\'t know the answer, you must say "I don\'t know".{SYSTEM_PROMPT}'),
        ('system', f'Answer the user\'s questions in the same language they use, primarily **Traditional Chinese (zh-tw)** or **English**, based on the context provided below: \n\n{{context}}\n If you don\'t know the answer, you must say "I don\'t know".{SYSTEM_PROMPT}'),
        ('user', 'Question: {input}'),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    context = retriever.invoke(query) # get_relevant_documents 更新

    # 呼叫 retrieval_chain 並取得結果
    response = retrieval_chain.invoke({
        'input': query,
        'context': context
    })
    # print(f"Query: {query}")
    # print(f"Answer: {response['answer']}")

    rag_res["answer"] = response["answer"]

    # 取得來源文件
    rag_res["source_documents"] = response["context"]
    rag_res["sources"] = [doc.page_content if isinstance(doc, Document) else str(doc) for doc in rag_res["source_documents"]]
    rag_res["metadata"] = [doc.metadata if isinstance(doc, Document) else str(doc) for doc in rag_res["source_documents"]]


    return rag_res

# ✅ 建立 RAG Tool
rag_tool = Tool(
    name="RAG_Search",
    func=query_rag_tool,
    description="Retrieves relevant documents using Vertex AI Vector Search and answers queries."
)

def get_rag_tools():
    return [rag_tool] if rag_tool else []

if __name__ == "__main__":
    query = " Apple in 2021 Q2 法說會議說了什麼？"
    result = rag_tool.run(query)
    print(result["answer"])
    print(result["sources"])
    uni = set(tuple(sorted(metadata.items())) for metadata in result["metadata"])
    print(uni, len(uni))
    txt_path = "/home/yang/Documents/TSMC/bsid_user08_tsmc/Transcript File/"
    transcript_filename = "Apple Inc. (NASDAQ AAPL) Q3 2021 Earnings Conference Call"
    if not transcript_filename.lower().endswith(".txt"):
            transcript_filename += ".txt"
    transcript_file_path = os.path.join(txt_path, transcript_filename)
    print(transcript_file_path)

    if os.path.exists(transcript_file_path):
        with open(transcript_file_path, mode="r", encoding="utf-8") as file:
            contents = file.read()

    all_match = all(content in contents for content in result["sources"])    
    print(all_match)