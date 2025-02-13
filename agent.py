from langgraph.graph import StateGraph
from typing import List, TypedDict
import json
from langchain_core.messages import HumanMessage
from sql_search import get_sql_tools
from rag_search import get_rag_tools
# from tool_selection import decide_tools
from langchain_google_vertexai import VertexAI
from langchain.chat_models import init_chat_model
from prompt import LLM_SQL_SYS_PROMPT, USER_DECIDE_SEARCH_PROMPT, MULTI_RAG_PROMPT
from config import (PROJECT_ID, REGION, BUCKET, INDEX_ID, 
                    ENDPOINT_ID, BUCKET_URI, 
                    MODEL_NAME, EMBEDDING_MODEL_NAME, MODEL_PROVIDER
                    )

def decide_tools(llm, query: str):
    """根據 Query 選擇 SQL 或 RAG 工具"""
    lower_q = query.lower()

    prompt = USER_DECIDE_SEARCH_PROMPT.format(query=lower_q)

    result = llm.invoke(prompt).content
    start = result.find("{")
    end = result.rfind("}") + 1
    json_part = result[start:end]

    # 手動解析
    search_types = eval(json_part.split('"search_types":')[1].split("]")[0].strip().strip(":").strip() + "]")
    print("Search Types:", search_types)

    return search_types


class AgentState(TypedDict):
    query: str
    adjusted_query: str
    tools: List[str]
    tool_results: List[str]
    final_answer: str  # 這裡的 key 改為 final_answer 避免衝突


class Agent:
    def __init__(self, model, sql_tools, rag_tools):
        self.model = model
        self.tools = sql_tools + rag_tools  
        self.tool_dict = {t.name: t for t in self.tools}

        # LangGraph: 建立 StateGraph
        graph = StateGraph(state_schema=AgentState)
        graph.add_node("decide", self.decide_action)
        graph.add_node("adjust_sql_query", self.adjust_sql_query)  # ✅ 新增 SQL 調整 Node
        graph.add_node("adjust_rag_query", self.adjust_rag_query)  # ✅ 新增 SQL 調整 Node
        graph.add_node("sql_action", self.take_action_sql)
        graph.add_node("rag_action", self.take_action_rag)
        graph.add_node("action", self.take_action)
        graph.add_node("generate_final_response", self.generate_final_response)  # ✅ 修正名稱避免衝突
        graph.add_node("end", lambda state: state)

        # 設定決策流程
        graph.add_conditional_edges(
            "decide",
            lambda state: "adjust_sql_query" if (self.is_sql_query(state) or self.is_rag_query(state)) else "action"
        )
        graph.add_edge("adjust_sql_query", "sql_action")
        graph.add_edge("sql_action", "adjust_rag_query")
        graph.add_edge("adjust_rag_query", "rag_action")
        graph.add_conditional_edges(
            lambda state: "rag_action" if (self.is_sql_query(state) or self.is_rag_query(state)) else "action",
            "generate_final_response"
        )
        # graph.add_edge("action", "generate_final_response")  # ✅ 修正名稱
        graph.add_edge("generate_final_response", "end")

        graph.set_entry_point("decide")
        self.graph = graph.compile()

    def decide_action(self, state: AgentState) -> AgentState:
        """決定應該使用哪些工具"""
        query = state["query"]
        selected_tools = decide_tools(self.model, query)
        if not selected_tools:
            return {"query": query, "adjusted_query": "", "tools": [], "tool_results": [], "final_answer": "很抱歉，目前沒有對應的資料。"}
        return {"query": query, "adjusted_query": "", "tools": selected_tools, "tool_results": [], "final_answer": ""}

    
    def is_sql_query(self, state: AgentState) -> bool:
        """檢查是否需要 Call SQL tool 調整"""
        return "sql_db_query" in state["tools"]
    
    def is_rag_query(self, state: AgentState) -> bool:
        """檢查是否需要 Call SQL tool 調整"""
        return "RAG_Search" in state["tools"]
    
    def fix_rag_result(self, text):
            
        text = text.replace("`", '"').replace("json", "")
        company_name_start = text.find('"Company Name": [') + len('"Company Name": [')
        company_name_end = text.find("]", company_name_start)
        company_names = text[company_name_start:company_name_end].replace('"', '').split(", ")

        calendar_year_start = text.find('"CALENDAR_YEAR": [') + len('"CALENDAR_YEAR": [')
        calendar_year_end = text.find("]", calendar_year_start)
        calendar_years = text[calendar_year_start:calendar_year_end].replace('"', '').split(", ")

        calendar_qtr_start = text.find('"CALENDAR_QTR": [') + len('"CALENDAR_QTR": [')
        calendar_qtr_end = text.find("]", calendar_qtr_start)
        calendar_qtrs = text[calendar_qtr_start:calendar_qtr_end].replace('"', '').split(", ")

        # ✅ 取得 Multiple Values Exist
        multiple_values_start = text.find('"Multiple Values Exist": "') + len('"Multiple Values Exist": "')
        multiple_values_end = text.find('"', multiple_values_start)
        multiple_values_exist = text[multiple_values_start:multiple_values_end]

        # ✅ 取得 Split Queries
        queries_start = text.find('"Split Queries": [') + len('"Split Queries": [')
        queries_end = text.find("]", queries_start)
        queries = text[queries_start:queries_end].replace('"', '').split(",\n        ")

        print(" - Company Name:", company_names)
        print(" - CALENDAR_YEAR:", calendar_years)
        print(" - CALENDAR_QTR:", calendar_qtrs)
        print("\nMultiple Values Exist:", multiple_values_exist)
        print("\nSplit Queries:")
        for q in queries:
            print(" -", q.strip())
        return {"Company Name": company_names, "CALENDAR_YEAR": calendar_years, "CALENDAR_QTR": calendar_qtrs, "Multiple Values Exist": multiple_values_exist, "Split Queries": queries}

    def adjust_sql_query(self, state: AgentState) -> AgentState:
        """調整 SQL Query，使其更加明確"""
        if not self.is_sql_query(state):
            return state
        
        query = state["query"]
        prompt = LLM_SQL_SYS_PROMPT.format(query=query)
        adjusted_query = self.model.invoke(prompt).content
        print(f"Adjusted SQL query: {adjusted_query}\n")
        return {"query": query, "adjusted_query": adjusted_query, "tools": state["tools"], "tool_results": [], "final_answer": ""}
 

    def adjust_rag_query(self, state: AgentState) -> AgentState:
        """調整 RAG Query，使其更加明確"""
        if not self.is_rag_query(state):
            return state
        
        query = state["query"]
        prompt = MULTI_RAG_PROMPT.format(query=query)
        is_multi_rag = self.model.invoke(prompt)
        multi_rag = self.fix_rag_result(is_multi_rag.content)

        adjusted_query = ""
        for each_query in multi_rag["Split Queries"]:

            adjusted_query += each_query+'\n'
        
        print(f"Adjusted SQL query: {adjusted_query}\n")
        return {"query": query, "adjusted_query": adjusted_query, "tools": state["tools"], "tool_results": [], "final_answer": ""}

    def take_action_sql(self, state: AgentState) -> AgentState:
        """執行查詢工具"""
        if not self.is_sql_query(state):
            return state
        
        query = state["adjusted_query"]
        tool_names = state["tools"]
        results = state["tool_results"]
        
        name = "sql_db_query"
        tool = self.tool_dict.get(name)
            # user_query = SQL_SYS_PROMPT + query
        tool_result = tool.run(query)
        results.append(f"{name} tool reponse : {tool_result['structured_response']}")
            
        print(f"--- {name} tool results:", results)
        return {"query": state["query"], "adjusted_query": query, "tools": tool_names, "tool_results": results, "final_answer": ""}
    
    def take_action_rag(self, state: AgentState) -> AgentState:
        """執行查詢工具"""
        if not self.is_rag_query(state):
            return state
        
        queries = state["adjusted_query"]
        tool_names = state["tools"]
        results = state["tool_results"]
        name = "RAG_Search"
        tool = self.tool_dict.get(name)

        for query in queries.split('\n'):
            
            # user_query = SQL_SYS_PROMPT + query
            tool_result = tool.run(query)
            results.append(f"{name} tool reponse : {tool_result['structured_response']}")
            
        print(f"--- {name} tool results:", results)
        return {"query": state["query"], "adjusted_query": queries, "tools": tool_names, "tool_results": results, "final_answer": ""}
    
    def take_action(self, state: AgentState) -> AgentState:
        """執行查詢工具"""
        query = state["query"]
        tool_names = state["tools"]
        results = state["tool_results"]
        name = "RAG_Search"
        tool = self.tool_dict.get(name)
        
        result = llm.invoke(query).content
            
        print("--- tool results:", result)
        return {"query": state["query"], "adjusted_query": query, "tools": tool_names, "tool_results": results, "final_answer": ""}

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

llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)

sql_tool = get_sql_tools()
rag_tool = get_rag_tools()
print("SQL Tools:", sql_tool, "\nRAG Tools:", rag_tool)
agent = Agent(model=llm, sql_tools=sql_tool, rag_tools=rag_tool)

if __name__ == "__main__":
    # 建立 Agent 物件
    llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
    agent = Agent(model=llm, sql_tools=get_sql_tools(), rag_tools=get_rag_tools())

    test_queries = [
        "What is Amazon's Revenue in 2022 Q1?",       # show 單一公司單一指標
        "show the `evenue`, `Tax Expense`, `Operating Expense` of Amazon in 2020 Q1.",     # show 單一公司多個指標
        "show the `evenue`, `Taxespense`, `Operating Expense` of Amazon in 2020 Q1.",     # show 單一公司多個指標, 但語法錯誤
        "show the all value of Amazon in 2020 Q1?",     # show 單一公司所有指標
        # "Find Amazon's 2020 Q1 earnings call insights.",
        # "Show both 2020 Amazon Q1 revenue and future outlook."
    ]   
    sigle_value_query = [
        "What is Amazon's Revenue in 2022 Q1?",
        "What is AMD's Operating Income in 2023 Q3?",
    ]
    trend_anlysis_query = [
        "Did Intel's Gross Profit Margin increase in 2020 Q4?",
        "Did Samsung's Operating Expense decrease in 2020 Q3?",
    ]
    
    time_based_comparison_query = [
        "Is Qualcomm's Total Asset in 2023 Q1 higher than in 2022 Q1?",
        "Is TSMC's Operating Margin in 2021 Q2 higher than in 2020 Q2?",
        "Is Microsoft's Tax Expense in 2024 Q1 lower than in 2023 Q4?",
        "Is Google's Revenue in 2022 Q2 higher than in 2021 Q2?",
        "Is Apple's Operating Income in 2021 Q1 higher than in 2020 Q1?",
        "Was Broadcom's Cost of Goods Sold lower in 2020 Q2 compared to Q1?",
    ]
    
    # single value query
    print("Single Value Query:")
    for query in sigle_value_query:
        print("="*50)
        print(f"Query: {query}")
        result = agent.run(query)
        print("Final Response:", result.content)
        
    # trend analysis query
    print("Trend Analysis Query:")
    for query in trend_anlysis_query:
        print("="*50)
        print(f"Query: {query}")
        result = agent.run(query)
        print("Final Response:", result.content)
        
    # time based comparison query
    print("Time Based Comparison Query:")
    for query in time_based_comparison_query:
        print("="*50)
        print(f"Query: {query}")
        result = agent.run(query)
        print("Final Response:", result.content)
    
    print("Done!")