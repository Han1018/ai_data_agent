from langgraph.graph import StateGraph
from typing import List, TypedDict
from langchain_core.messages import HumanMessage
from sql_search import get_sql_tools
from rag_search import get_rag_tools
from tool_selection import decide_tools
from langchain_google_vertexai import VertexAI
from langchain.chat_models import init_chat_model
from prompt import LLM_SQL_SYS_PROMPT
from config import (PROJECT_ID, REGION, BUCKET, INDEX_ID, 
                    ENDPOINT_ID, BUCKET_URI, 
                    MODEL_NAME, EMBEDDING_MODEL_NAME, MODEL_PROVIDER
                    )


class AgentState(TypedDict):
    query: str
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
        selected_tools = decide_tools(query)
        if not selected_tools:
            return {"query": query, "tools": [], "tool_results": [], "final_answer": "很抱歉，目前沒有對應的資料。"}
        return {"query": query, "tools": selected_tools, "tool_results": [], "final_answer": ""}

    def is_sql_query(self, state: AgentState) -> bool:
        """檢查是否需要 Call SQL tool 調整"""
        return "sql_db_query" in state["tools"]
    
    def adjust_sql_query(self, state: AgentState) -> AgentState:
        """調整 SQL Query，使其更加明確"""
        query = state["query"]
        adjusted_query = self.model.invoke(f"請將以下查詢轉換成具體且明確的 user query: {query}\n" + LLM_SQL_SYS_PROMPT).content
        
        print(f"Adjusted SQL query: {adjusted_query}\n")
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
                
            print(f"--- {name} tool results:", results)
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