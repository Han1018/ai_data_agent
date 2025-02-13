from vertexai import init
from sqlalchemy import create_engine
from langchain.chat_models import init_chat_model
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from config import DB_HOST, DB_PORT, DATABASE, _USER, _PASSWORD, PROJECT_ID, REGION, MODEL_NAME, MODEL_PROVIDER


init(project=PROJECT_ID, location=REGION)
llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)

# ✅ 建立 SQLAlchemy 連線
db_url = f'postgresql+psycopg2://{_USER}:{_PASSWORD}@{DB_HOST}:{DB_PORT}/{DATABASE}'
engine = create_engine(db_url)

# ✅ 創建 SQL Database 工具
db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # llm 由 agent.py 負責初始化
tools = toolkit.get_tools()

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="PostgreSQL", top_k=5)

# init langGraph
agent_executor = create_react_agent(llm, tools, prompt=system_message)

# ✅ 建立 SQL 查詢工具
def sql_query_tool(query: str) -> dict:
    """透過 SQL Agent 生成 SQL 並執行，並回傳包含 structured_response 的 dict"""
    # 使用 invoke 並傳入正確格式的輸入（字典格式的 state）
    response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    
    # 提取最終答案 (根據實際回傳結構調整)
    if "messages" in response:
        output = response["messages"][-1].content
    else:
        output = str(response)
    
    return {"structured_response": output}

sql_tool = Tool(
    name="sql_db_query",
    func=sql_query_tool,
    description="用來查詢 SQL 數據庫，請輸入財務相關的問題，系統會自動轉換為 SQL 語句並執行。"
)

def get_sql_tools():
    return [sql_tool]






# 測試
if __name__ == "__main__":
    question = "show 5 first rows in the `fin_data` table."

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()