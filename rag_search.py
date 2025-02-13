import os
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings, VectorSearchVectorStore
from langchain.chains import RetrievalQA
from vertexai.preview.generative_models import GenerativeModel
from langchain.tools import Tool
import re
from langchain_core.documents import Document
from config import (PROJECT_ID, REGION, BUCKET, INDEX_ID, 
                    ENDPOINT_ID, BUCKET_URI, 
                    MODEL_NAME, EMBEDDING_MODEL_NAME
                    )
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)
from prompt import RAG_SPLIT_QUERY_PROMPT

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

def update_filters(company_name, calendar_year, calendar_qtr):
    """更新篩選條件"""
    filters = []
    numeric_filters = []
    if company_name:
        filters.append(Namespace(name="Company Name", allow_tokens=[company_name]))
    if calendar_year:
        numeric_filters.append(NumericNamespace(name="CALENDAR_YEAR", value_float=float(calendar_year), op="EQUAL"))
    if calendar_qtr:
        filters.append(Namespace(name="CALENDAR_QTR", allow_tokens=[calendar_qtr]))

    return filters, numeric_filters

def query_rag_tool(query: str):
    """使用 Vertex AI Vector Search 進行檢索，並使用 Gemini 1.5 Pro 解析 Query"""
    llm = VertexAI(model_name=MODEL_NAME)

    # 🔍 使用 Gemini 解析 Query，提取資訊
    company_name, calendar_year, calendar_qtr = extract_info_from_query(llm, query)

    print(f"Extracted Info:\n - Company Name: {company_name}\n - CALENDAR_YEAR: {calendar_year}\n - CALENDAR_QTR: {calendar_qtr}")

    # 📌 設定檢索條件（如果有）
    filters, numeric_filters = update_filters(company_name, calendar_year, calendar_qtr)

    print(filters, numeric_filters)
    retriever.search_kwargs = {
    "k": 10,
    "filter": filters if filters else None,  # ✅ 確保 `filter` 只出現一次
    "numeric_filter": numeric_filters if numeric_filters else None,  # ✅ 正確加入數值篩選
    }

    # 🔍 進行檢索並回答
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    response = retrieval_qa({"query": query})
    result = response["result"]

    # 取得來源文件
    source_docs = response["source_documents"]
    sources = [doc.page_content if isinstance(doc, Document) else str(doc) for doc in source_docs]
    metadata_list = [doc.metadata if isinstance(doc, Document) else str(doc) for doc in source_docs]


    return {
        "answer": result,
        "sources": sources,
        "metadata": metadata_list,
        "source_documents": source_docs,
        "extracted_info": {
            "Company Name": company_name,
            "CALENDAR_YEAR": calendar_year,
            "CALENDAR_QTR": calendar_qtr,
        }
    }

# ✅ 建立 RAG Tool
rag_tool = Tool(
    name="RAG_Search",
    func=query_rag_tool,
    description="Retrieves relevant documents using Vertex AI Vector Search and answers queries."
)

def get_rag_tools():
    return [rag_tool]


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