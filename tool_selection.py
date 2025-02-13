
def decide_tools(query: str):
    """根據 Query 選擇 SQL 或 RAG 工具"""
    lower_q = query.lower()

    numeric_keywords = ["revenue", "operating income", "sql", "fin_data"]
    text_keywords = ["法說", "meeting", "transcript", "rag"]

    need_sql = any(k in lower_q for k in numeric_keywords)
    need_rag = any(k in lower_q for k in text_keywords)


    
    if need_sql and need_rag:
        return ["sql_db_query", "RAG_Search"]
    elif need_sql:
        return ["sql_db_query"]
    elif need_rag:
        return ["RAG_Search"]
    return []