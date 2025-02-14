from sqlalchemy import create_engine
import sqlalchemy
from langchain_community.utilities.sql_database import SQLDatabase
from vertexai import init
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.tools import Tool
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import vertexai
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQA
from typing import List, TypedDict
from langgraph.graph import StateGraph
from pydantic import Field
from typing import Any

from langchain_google_vertexai import (
    VertexAIEmbeddings,
    VectorSearchVectorStore,
)

from prompt import LLM_SQL_SYS_PROMPT, LLM_SQL_CHECK_COMPANY, CLEAN_FORMAT
from IPython.display import Image, display

import json
import re
from langchain.schema import AIMessage

ACCESSIBLE_COMPANIES_LIST = {
    "cn" : ["Baidu", "Tencent"],
    "kr" : ["Samsung"],
    "gb" : ["all"]
}

ROLE = "cn"  # 設定使用者角色
############################################
# 1. 設定資料庫連線
############################################
PROJECT_ID = "tsmccareerhack2025-bsid-grp2"
REGION = "us-central1"  
INSTANCE = "sql-instance-relational"
DATABASE = "postgres" 
TABLE_NAME = "fin_data" 
DB_HOST = "34.56.145.52"  # Cloud SQL Public IP
DB_PORT = "5432"  # PostgreSQL 預設端口
_USER = "postgres"
_PASSWORD = "postgres"

db_url = f'postgresql+psycopg2://{_USER}:{_PASSWORD}@{DB_HOST}:{DB_PORT}/{DATABASE}'
engine = sqlalchemy.create_engine(db_url)
db = SQLDatabase(engine)


############################################
# 2. 初始化 Vertex AI & LLM
############################################
init(project=PROJECT_ID, location=REGION)
llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai")
cot_llm = init_chat_model("gemini-2.0-flash-thinking-exp-01-21", model_provider="google_vertexai")

############################################
# 3. 設定 SQL Agent 工具
############################################
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="PostgreSQL", top_k=5)
agent_executor = create_react_agent(llm, tools, prompt=system_message)


def extract_sql_from_response(response):
    """從 LLM 回傳的內容中提取 SQL 查詢"""
    if "messages" in response:
        return response["messages"][-1].content.strip()
    return str(response).strip()

class SQLQueryGenerator(Tool):
    """覆寫 SQLDatabaseToolkit，讓 LLM 只產生 SQL，而不執行"""

    agent_executor: Any = Field(...)  # 明確宣告 agent_executor，讓 pydantic 允許這個屬性

    def __init__(self, agent_executor, **kwargs):
        super().__init__(
            name="Generate SQL",
            description="Generate SQL query based on user input without executing it.",
            func=self.run,  # 🛠 **這裡修正，指定一個可執行函式**
            agent_executor=agent_executor,  # 傳入 `agent_executor`
            **kwargs
        )

    # 產生 SQL 語法
    def run(self, query):
        """讓 LLM 產生 SQL 查詢但不執行"""
        response = self.agent_executor.invoke({"messages": [HumanMessage(content=query)]})

        db_queries = []
        
        # 檢查 response["messages"] 內的 `tool_calls`
        for message in response["messages"]:
            if isinstance(message, AIMessage) and "function_call" in message.additional_kwargs:
                function_call = message.additional_kwargs["function_call"]

                if function_call["name"] == "sql_db_query":
                    try:
                        # 解析 JSON 字符串
                        function_args = json.loads(function_call["arguments"])
                        sql_query = function_args["query"]
                        db_queries.append(sql_query)
                    except json.JSONDecodeError as e:
                        print("JSON 解析錯誤: ", e)

            # 如果 `tool_calls` 是獨立陣列
            if "tool_calls" in message.additional_kwargs:
                for tool_call in message.additional_kwargs["tool_calls"]:
                    if tool_call["name"] == "sql_db_query":
                        sql_query = tool_call["args"]["query"]
                        db_queries.append(sql_query)

        # print("db_queries: ", db_queries)
        
        if db_queries:
            return db_queries[0]  # 只取第一個 SQL 查詢
        return "No SQL query generated"

def modify_query_with_companies(query: str, allow_companies: list) -> str:
    """使用 LLM 生成修改後的 SQL 查詢，確保只存取允許的公司"""

    # 如果允許所有公司，則直接返回原始查詢
    if allow_companies == ['all']:
        return query

    # 格式化公司名稱以符合 SQL 語法
    company_filter = ", ".join([f"'{company}'" for company in allow_companies])

    # 使用 LLM 生成新的 SQL 查詢
    new_sql = cot_llm.invoke(LLM_SQL_CHECK_COMPANY.format(sql_query = query, company_filter = company_filter)).content
    
    # 清理多餘不相關語句
    clean_prompt =CLEAN_FORMAT.format(user_input = new_sql)
    clen_sql = llm.invoke(clean_prompt+new_sql).content
    
    return clen_sql


sql_generator = SQLQueryGenerator(agent_executor)  # 創建 SQL 產生工具

def sql_query_tool(query: str) -> dict:
    """生成 SQL 查詢並執行回覆 query question"""
    
    # 讓 LLM 產生 SQL 查詢
    generated_sql = sql_generator.run(query)

    # 在查詢中加入權限控制條件
    accessiable_company = ACCESSIBLE_COMPANIES_LIST[ROLE]
    secured_sql = modify_query_with_companies(generated_sql, accessiable_company)

    # 執行修改後的 SQL
    try:
        db_response = db.run(secured_sql)
    except Exception as e:
        print(f"SQL query failed: {e}")
        db_response = f"No data found for query."
    
    # 合併 user question, db query, db response
    query_with_db_result = f"User question : {query} \nDB query : {secured_sql}\nDB response : {db_response}。\n\n 如果 DB query = `SELECT * FROM fin_data WHERE 1 = 0;` 請回答資料庫中沒有找到相關的資料，可能是沒有相關的資料或是權限不足。"
    
    # llm 根據 query & db answer 生成回應
    final_prompt = "Please reponse to user, reponse the user's question and the database result. USD:TWD = 1 : 32.93 if user question need.\n \
    Below is the user question and database query result: \n"
    sql_query_res = cot_llm.invoke(final_prompt + query_with_db_result).content
    
    return {"structured_response": sql_query_res}

# 創建 Tool 物件
sql_tool = Tool(
    name="sql_db_query",
    func=sql_query_tool,
    description="用來查詢 SQL 數據庫，請輸入財務相關的問題，系統會自動轉換為 SQL 語句並執行。"
)
# 更新 sql_tools，確保包含新的 `sql_tool`
sql_tools = [sql_tool]

############################################
# 4. 設定 RAG 工具
############################################
BUCKET = "tsmccareerhack2025-bsid-grp2-bucket"
BUCKET_URI = f"gs://{BUCKET}"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
llm_model = VertexAI(model_name="gemini-1.5-pro")
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

# 設定 Vertex AI Vector Search
INDEX_ID = "4438785615936356352"
ENDPOINT_ID = "2966706672111714304"
my_index = aiplatform.MatchingEngineIndex(INDEX_ID)
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT_ID)

# 建立向量資料庫
vector_store = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=REGION,
    gcs_bucket_name=BUCKET,
    index_id=my_index.name,
    endpoint_id=my_index_endpoint.name,
    embedding=embedding_model,
)

# 建立 RetrievalQA Chain
retriever = vector_store.as_retriever()
retriever.search_kwargs = {"k": 10}
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# 建立 RAG tool
def query_rag_tool(query: str):
    """使用 Vertex AI Vector Search 進行檢索，並使用 Gemini 1.5 Pro 回答"""
    response = retrieval_qa({"query": query})
    return {"answer": response["result"], "sources": [doc.page_content for doc in response["source_documents"]]}

rag_tool = Tool(
    name="RAG_Search",
    func=query_rag_tool,
    description="Retrieves relevant documents using Vertex AI Vector Search and answers queries."
)
rag_tools = [rag_tool]

############################################
# 5. 建立 LangGraph Agent
############################################
class AgentState(TypedDict):
    query: str
    tools: List[str]
    tool_results: List[str]
    final_answer: str 

def decide_tools(llm, query: str) -> List[str]:
    """
    根據 query，決定要使用哪個 tool。
    如果包含數值型關鍵字 → SQL
    如果包含文本/法說關鍵字 → RAG
    可能同時需要兩者 → 皆用
    如果都沒有 → 回傳空 list
    """
    lower_q = query.lower()

    # 這些是你在圖片裡提到的財報指標，或 fin_data 等關鍵字
    numeric_keywords = [
        "operating income", "cost of goods sold", "operating expense",
        "tax expense", "revenue", "total asset",
        "gross profit margin", "operating margin", "fin_data", "sql", "table"
    ]
    # 這些則是假設要查「法說會議」或其它文字檔用的關鍵字
    text_keywords = ["法說", "會議", "rag", "meeting", "transcript", "txt"]

    need_sql = any(k.lower() in lower_q for k in numeric_keywords)
    need_rag = any(k.lower() in lower_q for k in text_keywords)

    # 依照需求可能同時都要
    if need_sql and need_rag:
        return ["sql_db_query", "RAG_Search"]
    elif need_sql:
        return ["sql_db_query"]
    elif need_rag:
        return ["RAG_Search"]
    else:
        # 都不符合
        return []
    

class Agent:
    def __init__(self, model, sql_tools, rag_tools):
        self.model = model
        self.tools = sql_tools + rag_tools  
        self.tool_dict = {t.name: t for t in self.tools}

        # LangGraph: 建立 StateGraph
        graph = StateGraph(state_schema=AgentState)
        graph.add_node("decide", self.decide_action)
        graph.add_node("adjust_sql_query", self.adjust_sql_query)  # ✅ 新增 SQL 調整 Node
        graph.add_node("action", self.take_action)
        graph.add_node("generate_final_response", self.generate_final_response)  # ✅ 修正名稱避免衝突
        graph.add_node("end", lambda state: state)

        # 設定決策流程
        # graph.add_edge("decide", "action")
        graph.add_conditional_edges(
            "decide",
            lambda state: "adjust_sql_query" if self.is_sql_query(state) else "action"
        )
        graph.add_edge("adjust_sql_query", "action")
        graph.add_edge("action", "generate_final_response")  # ✅ 修正名稱
        graph.add_edge("generate_final_response", "end")

        graph.set_entry_point("decide")
        self.graph = graph.compile()

    def decide_action(self, state: AgentState) -> AgentState:
        """決定應該使用哪些工具"""
        query = state["query"]
        selected_tools = decide_tools(self.model, query)
        if not selected_tools:
            return {"query": query, "tools": [], "tool_results": [], "final_answer": "很抱歉，目前沒有對應的資料。"}
        return {"query": query, "tools": selected_tools, "tool_results": [], "final_answer": ""}

    def is_sql_query(self, state: AgentState) -> bool:
        """檢查是否需要 Call SQL tool 調整"""
        return "sql_db_query" in state["tools"]
    
    def adjust_sql_query(self, state: AgentState) -> AgentState:
        """調整 SQL Query，使其更加明確"""
        query = state["query"]
        query_and_prompt = LLM_SQL_SYS_PROMPT.format(user_query=query)
        adjusted_query = self.model.invoke(query_and_prompt).content
        
        return {"query": adjusted_query, "tools": state["tools"], "tool_results": [], "final_answer": ""}

    def take_action(self, state: AgentState) -> AgentState:
        """執行查詢工具"""
        query = state["query"]
        tool_names = state["tools"]
        results = []
        
        for name in tool_names:
            tool = self.tool_dict.get(name)
            if name in ["sql_db_query"]:
                # user_query = SQL_SYS_PROMPT + query
                tool_result = tool.run(query)
                results.append(f"{name} tool reponse : {tool_result['structured_response']}")
                
            elif name in ["RAG_Search"]:
                tool_result = tool.run(query)
                results.append(f"=== Tool {name} 回傳 ===\n{tool_result}")
            else:
                results.append(f"無法找到對應的 tool: {name}")
                
        return {"query": query, "tools": tool_names, "tool_results": results, "final_answer": ""}

    def generate_final_response(self, state: AgentState) -> AgentState:
        """將 tools 查詢結果與 user 問題整合，再交給 LLM 重新回答"""
        query = state["query"]
        tool_results = "\n".join(state["tool_results"])

        prompt = f"""
        使用者的問題: {query}
        以下是來自不同工具的查詢結果：
        {tool_results}
        ---
        請根據使用者問題的語言回覆, 英文問問題請用英文回答, 以此類推。
        請根據這些資訊，產生一個完整且清楚的回答，並確保你的回答能讓使用者理解。 
        """
        # print('final round query:', prompt)
        
        final_answer = self.model.invoke(prompt)
        return {"query": query, "tools": state["tools"], "tool_results": state["tool_results"], "final_answer": final_answer}

    def run(self, query: str) -> str:
        """對外的介面，餵入 query 後跑 graph，回傳最後 response"""
        init_state: AgentState = {"query": query, "tools": [], "tool_results": [], "final_answer": ""}
        end_state = self.graph.invoke(init_state)
        return end_state["final_answer"]
    
if __name__ == "__main__":
    # 建立 Agent 物件
    agent = Agent(model=llm, sql_tools=sql_tools, rag_tools=rag_tools)

    failed_queries = [
        "Did Intel's Gross Profit Margin increase in 2020 Q4?"
    ]
        
    test_data = {
        # "new" : [
        #     "What is Samsung's gross profit margin in 2023 Q3?"
        # ],
        
        # "twd currency": [
        #     # "What was the Revenue in Q1 2020 for Amazon in TWD?",
        #     # "Samsung 2020~2024 Q2 的 Revenue 是多少 USD?",
        # ],
        
        # "annual_query": [
        #     "Samsung 2020〜2024 Q1 的 Revenue 是否持續增長？",
        #     "Samsung 2021〜2023 Q3 的 Operating Margin 是否穩定？"
        # ],
        "Quarterly_Comparisons": [
            "Baidu 2020 Q4 和 2023 Q4 的 Revenue 變化幅度是多少？",
            "TSMC 2021 Q1 與 2021 Q2 的 Operating Income 哪一季較高？",
            # "Google 2021 Q4 和 2022 Q4 的 Revenue 變動幅度是多少？",
        ],
        # "Complex_Questions":[
        #     "Intel 2020 Q4 和 2023 Q4 的 Gross Profit Margin 是否上升？",
        #     "Samsung 2021 Q3 和 2023 Q3 的 Operating Expense 是否增加？",
        #     "TSMC 2021 Q1 和 2024 Q1 的 Operating Margin 變化是否與 Samsung 相同？",
        #     "Google, Apple, Microsoft 2020〜2024 Q4 的 Revenue 誰增幅最大？",
        #     "Intel 2023 Q2 的 Operating Expense 是否當季特別高？",
        #     "Amazon 2021 Q4 的 Tax Expense 是否對比前季異常升高？"
        # ]
    }
    
    print("=== start test data ===")
    for data in test_data.values():
        for query in data:
            print(f"Query: {query}")
            print(f"Final Answer:\n"+agent.run(query).content)
            print("===")