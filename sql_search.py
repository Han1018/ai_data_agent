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
from prompt import LLM_IS_IN_ALLOW_COMPANY, CHECK_CAN_DRAW_PROMPT, GEN_DRAW_IMAGE_FORMAT_PROMPT, GET_Yes_No_PROMPT, \
    IMAGE_INFO_PROMPT, HAS_MULTI_COMPANY_PROMPT, CHECK_CAN_DRAW_HISTOGRAM_PROMPT, GEN_DRAW_HISTOGRAM_FORMAT_PROMPT, \
        IMAGE_HIST_INFO_PROMPT
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
import base64
import io
import json

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pdb
from collections import defaultdict

# 設定字型檔案的絕對路徑
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 確保字型存在
if not os.path.exists(font_path):
    raise FileNotFoundError(f"字型檔案未找到: {font_path}")

# 建立字型物件
chinese_font = fm.FontProperties(fname=font_path)

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

def extract_json_info(input_str):
    # 去掉開頭和結尾的三個反引號，並且去除 `json` 部分
    cleaned_str = input_str.strip('`').replace('json', '').strip()
    
    try:
        data = json.loads(cleaned_str)  # 將 JSON 字串轉換為 Python 字典
        title = data.get("title", "圖表")  # 取得 title，若沒有則給預設值
        y_label = data.get("y_label", "Value")  # 取得 y_label，若沒有則給預設值
        return title, y_label
    except json.JSONDecodeError:
        print("無法解析 JSON 字串，請檢查格式。")
        return "圖表", "Value"  

def extract_json_info_xy_title(input_str):
    # 去掉開頭和結尾的三個反引號，並且去除 `json` 部分
    cleaned_str = input_str.strip('`').replace('json', '').strip()
    print("input_str: ", input_str)
    print("cleaned_str: ", cleaned_str)
    
    try:
        data = json.loads(cleaned_str)  # 將 JSON 字串轉換為 Python 字典
        title = data.get("title", "圖表")  # 取得 title，若沒有則給預設值
        y_label = data.get("y_label", "Value")  # 取得 y_label，若沒有則給預設值
        x_label = data.get("x_label", "Company Name")  # 取得 x_label，若沒有則給預設值
        return title, y_label, x_label
    
    except json.JSONDecodeError:
        print("無法解析 JSON 字串，請檢查格式。")
        return "圖表", "Value", "Company Name"
    
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

def extract_data_from_response(response):
    """從 LLM 回應中提取數據，支援 YYYY-QN 格式"""
    lines = response.split("\n")
    data = []
    
    # 修改正則表達式，允許 YYYY-QN 格式
    for line in lines:
        match = re.match(r"(\d{4}-Q[1-4]):\s*([\d\.]+)", line.strip())
        if match:
            date, value = match.groups()
            data.append((date, float(value)))  # 允許小數點數值
            
    return data

def extract_bar_data(text):
    """
    從 LLM 回應中提取分組直方圖的數據。

    參數:
    - text: 包含公司名稱與數值的文字，格式如下：
      ```
      Company A: 100 Company A: 120 Company A: 90 Company B: 110 Company B: 95
      ```

    回傳:
    - List[Tuple[str, int]] -> [(公司名稱, 數值), ...]
    """
    import re
    from collections import defaultdict

    # 提取三個反引號之後的內容
    match = re.search(r"```([\s\S]+?)```", text)
    if not match:
        print("無法找到有效的數據區塊")
        return None

    data_text = match.group(1).strip()

    # 解析數據
    data_dict = defaultdict(int)  # 改用 int 作為值的類型，而不是 list
    for match in re.finditer(r"([\w\s]+):\s*(\d+)", data_text):
        company, value = match.groups()
        data_dict[company.strip()] += int(value)  # 累加數值

    # 轉換成列表格式，確保格式為 (str, int)
    return sorted(data_dict.items())  # 讓公司名稱排序，確保畫圖時順序一致



def plot_bar_chart(data, title="公司指標比較", xlabel="公司", ylabel="數值", filename="bar_chart.png"):
    """
    繪製多家公司在單一指標上的直方圖，並儲存為本地圖片。

    參數:
    - data: List[Tuple[str, float]] -> [(公司名稱, 數值), ...]
    - title: 圖表標題
    - xlabel: X 軸標籤名稱
    - ylabel: Y 軸標籤名稱
    - filename: 儲存的圖片檔名

    回傳:
    - 儲存的圖片路徑
    """
    if not data:
        print("無法繪製圖表，因為沒有數據。")
        return None

    companies, values = zip(*data)  # 解析公司名稱與對應數值

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(companies, values, color="skyblue")

    ax.set_xlabel(xlabel, fontproperties=chinese_font)
    ax.set_ylabel(ylabel, fontproperties=chinese_font)
    ax.set_title(title, fontproperties=chinese_font)

    ax.set_xticklabels(companies, rotation=45, fontproperties=chinese_font)  # 確保 X 軸標籤使用中文字體
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 儲存圖片
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)  # 關閉圖表，避免記憶體浪費

    print(f"直方圖已成功儲存為: {os.path.abspath(filename)}")
    return filename  # 回傳圖片檔案名稱

def plot_line_chart(data, title="折線圖", x_label = "Time", y_label="Value", filename="chart.png"):
    """繪製折線圖並存為本地圖片"""
    if not data:
        print("無法繪製圖表，因為沒有數據。")
        return None

    dates, values = zip(*sorted(data))  # 確保按時間排序

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, values, marker="o", linestyle="-", color="b")

    ax.set_xlabel("Time", fontproperties=chinese_font)
    ax.set_ylabel(y_label, fontproperties=chinese_font)
    ax.set_title(title, fontproperties=chinese_font)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, fontproperties=chinese_font)
    ax.grid(True)

    # 儲存圖片
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)  # 關閉圖表，避免記憶體浪費

    print(f"圖表已成功儲存為: {os.path.abspath(filename)}")  # 顯示完整路徑
    return filename  # 回傳圖片檔案名稱

def sql_plot_bar_chart(db_response, title="公司指標比較", x_label="公司", y_label="數值", filename="bar_chart.png"):
    # 轉換 db response 為 matplotlib 格式
    prompt = GEN_DRAW_HISTOGRAM_FORMAT_PROMPT.format(sql_tool_result = db_response) # 將 db response 轉成 matplotlib 格式
    response = cot_llm.invoke(prompt).content.split('json',1)[0]
    print("response: ", response)
    
    # 提取畫圖數據
    data_points = extract_bar_data(response) # 提取數據
    print("data_points: \n", data_points)
    
    # 生成圖表標題和 XY 軸標籤
    # print("db_response: ", db_response)
    # prompt = IMAGE_HIST_INFO_PROMPT.format(user_input = db_response.strip())
    # response = llm.invoke(prompt).content.strip()
    # print("response: ", response)
    # title, y_label, x_label = extract_json_info_xy_title(response)
    
    try:
        plot_bar_chart(data_points, title=title, xlabel=x_label, ylabel=y_label, filename="bar_chart.png")
    except Exception as e:
        print(f"無法繪製圖表：{e}")
    
def sql_plot_line_chart(db_response, title="折線圖", x_label = "Time", y_label="Value", filename="chart.png"):
    prompt = GEN_DRAW_IMAGE_FORMAT_PROMPT.format(sql_tool_result = db_response) # 將 db response 轉成 matplotlib 格式
    response = cot_llm.invoke(prompt).content
    print("response: ", response)
    
    # 提取畫圖數據
    data_points = extract_data_from_response(response) # 提取數據
    print("data_points: \n", data_points)
    
    # 生成圖表標題和 Y 軸標籤
    # prompt = IMAGE_INFO_PROMPT.format(user_input = db_response)
    # response = llm.invoke(prompt).content.strip()
    # print("response: ", response)
    # title, y_label = extract_json_info(response)
    
    try:
        plot_line_chart(data_points, y_label=y_label, title=title, filename="chart.png")
    except Exception as e:
        print(f"無法繪製圖表：{e}")

class SQLQueryInput(BaseModel):
    user_question: str
    role: str

# ✅ 建立 SQL 查詢工具
def sql_query_tool(user_question: str, role: str) -> dict:
    """生成 SQL 查詢並執行回覆 query question"""
    
    role = role
    user_question = user_question
    # print("ACCESSIBLE COMPANIES: ", ACCESSIBLE_COMPANIES_LIST[role])
    
    # 檢查是否有權限查詢 db
    is_allow = is_in_accessable_companies(user_question, ACCESSIBLE_COMPANIES_LIST[role])
    print("IS ALLOW ACCESS DB: ", is_allow)
    
    if is_allow:
        response = agent_executor.invoke({"messages": [HumanMessage(content=user_question)]})
            # 提取最終答案 (根據實際回傳結構調整)
        if "messages" in response:
            db_response = response["messages"][-1].content
        else:
            db_response = str(response)
        print("SQL Tool response: ", db_response)
    
    else:
        db_response = "資料庫中沒有找到相關的資料，可能是沒有相關的資料或是權限不足。"
    
    # 確認 company 數量
    prompt = HAS_MULTI_COMPANY_PROMPT.format(db_response = db_response)
    reponse = cot_llm.invoke(prompt).content
    is_multi_company = llm.invoke(GET_Yes_No_PROMPT.format(user_input = reponse)).content.strip()
    print("is_multi_company:", is_multi_company)
    
    # 多 company
    if is_multi_company in ["y", "Y"]:
        prompt = CHECK_CAN_DRAW_HISTOGRAM_PROMPT.format(min_data_points = 2, sql_tool_result = db_response)
        reponse = cot_llm.invoke(prompt).content
        need_draw = llm.invoke(GET_Yes_No_PROMPT.format(user_input = reponse)).content.strip()
        print("need_draw:", need_draw)
        
        if need_draw in ["y", "Y"]:
            sql_plot_bar_chart(db_response, filename="bar_chart.png")
        else:
            print("不需要繪製圖表。")
    
    # 1 company
    else:
        # 檢查畫圖需求
        prompt = CHECK_CAN_DRAW_PROMPT.format(min_data_points = 3, sql_tool_result = db_response)
        reponse = cot_llm.invoke(prompt).content
        need_draw = llm.invoke(GET_Yes_No_PROMPT.format(user_input = reponse)).content.strip()
        print("need_draw:", need_draw)
        
        # 繪製圖表
        if need_draw in ["y", "Y"]:
            sql_plot_line_chart(db_response, filename="chart.png")
        else:
            print("不需要繪製圖表。")
        
    return {"structured_response": db_response, "file_name": "chart.png"}

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
    question = "Retreive Amazon, Apple, Samsung 2020 Q1 的 Revenue 資料。"

    # for step in agent_executor.stream(
    #     {"messages": [{"role": "user", "content": question}]},
    #     stream_mode="values",
    # ):
    #     step["messages"][-1].pretty_print()
    res = sql_query_tool(question, "GB")