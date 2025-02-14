from vertexai import init
from sqlalchemy import create_engine
from langchain.chat_models import init_chat_model
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel
from config import DB_HOST, DB_PORT, DATABASE, _USER, _PASSWORD, PROJECT_ID, REGION, MODEL_NAME, MODEL_PROVIDER, COT_LLM_MODEL_NAME
from prompt import LLM_IS_IN_ALLOW_COMPANY

ACCESSIBLE_COMPANIES_LIST = {
    "CN" : ["Baidu", "Tencent"],
    "KR" : ["Samsung"],
    "GB" : ["all"]
}

init(project=PROJECT_ID, location=REGION)
llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
cot_llm = init_chat_model(COT_LLM_MODEL_NAME, model_provider="google_vertexai")

# ✅ 建立 SQLAlchemy 連線
db_url = f'postgresql+psycopg2://{_USER}:{_PASSWORD}@{DB_HOST}:{DB_PORT}/{DATABASE}'
engine = create_engine(db_url)

# ✅ 創建 SQL Database 工具
db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # llm 由 agent.py 負責初始化
tools = toolkit.get_tools()

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="PostgreSQL", top_k=10)

# init langGraph
agent_executor = create_react_agent(llm, tools, prompt=system_message)

def is_in_accessable_companies(query: str, allow_companies: list) -> bool:
    if allow_companies == ['all']:
        return True
    
    # 格式化公司名稱以符合 SQL 語法
    company_filter = ", ".join([f"'{company}'" for company in allow_companies])
    not_allow = cot_llm.invoke(LLM_IS_IN_ALLOW_COMPANY.format(user_question = query, company_filter = company_filter)).content # only return y or n in string
    
    if not_allow in ["y", "Y"]:
        return False
    else:
        return True

class SQLQueryInput(BaseModel):
    user_question: str
    role: str

# ✅ 建立 SQL 查詢工具
def sql_query_tool(user_question: str, role: str) -> dict:
    """生成 SQL 查詢並執行回覆 query question"""
    
    role = role
    user_question = user_question
    # print("ACCESSIBLE COMPANIES: ", ACCESSIBLE_COMPANIES_LIST[role])
    
    is_allow = is_in_accessable_companies(user_question, ACCESSIBLE_COMPANIES_LIST[role])
    print("IS ALLOW: ", is_allow)
    
    if is_allow:
        response = agent_executor.invoke({"messages": [HumanMessage(content=user_question)]})
            # 提取最終答案 (根據實際回傳結構調整)
        if "messages" in response:
            db_response = response["messages"][-1].content
        else:
            db_response = str(response)
        # print("SQL Tool response: ", db_response)
    
    else:
        db_response = "資料庫中沒有找到相關的資料，可能是沒有相關的資料或是權限不足。"
        
    return {"structured_response": db_response}

# 使用 StructuredTool 來處理多個參數
sql_tool = StructuredTool(
    name="sql_db_query",
    func=sql_query_tool,
    description="用來查詢 SQL 數據庫，請輸入財務相關的問題，系統會自動轉換為 SQL 語句並執行。",
    args_schema=SQLQueryInput  # 這行讓 Tool 知道需要哪些參數
)

def get_sql_tools():
    return [sql_tool] if sql_tool else []






# 測試
if __name__ == "__main__":
    question = "show 5 first rows in the `fin_data` table."

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()