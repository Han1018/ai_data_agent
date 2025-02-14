LLM_SQL_SYS_PROMPT = """Transform user questiom into a more specific, complete, and easily interpretable question for the model. User quesition : {user_query}

Below is the database schema for a financial database. Please translate the user's question into a output example style question.

**Table Structure:**
- Table Name: fin_data
- Columns:
  - id (PRIMARY KEY, no analytical significance)
  - company_name (VARCHAR) - Company name
  - index (VARCHAR) - Financial metric, ENUM values:
    • `Operating Income`
    • `Cost of Goods Sold` 
    • `Operating Expense`
    • `Tax Expense`
    • `Revenue`
    • `Total Asset`
  - calendar_year (INT) - Fiscal year (YYYY format)
  - calendar_qtr (INT) - Fiscal quarter (1-4)
  - usd_value (NUMERIC) - Value in USD
  - local_currency (VARCHAR) - Original reporting currency
  - val_unit (VARCHAR) - Unit of measure (e.g., millions, thousands)
  - local_value (NUMERIC) - Value in original currency

- Output Guidelines: 
    1. Always filter 'index' using exact enum values
    2. Include currency units when presenting values
    3. Include USD and local currency in output. if not specified.
    4. The fin_data index should be selected from the enum after understanding the user's needs.
    5. Include the calendar_year and calendar_qtr if the user specifies a time period. 
    6. If the user specifies a financial metric, wrap the index name in backticks (``).
    7. Exchange rate of USD:TWD = 1 : 32.93

- Index information:
    - Operating Income = 營業利益
    - Cost of Goods Sold = 營業成本
    - Operating Expense = 營業費用
    - Tax Expense = 稅額
    - Revenue = 收入
    - Total Asset = 總資產
    - Gross profit margin(毛利率) = ((revenue 收入 - cost of goods sold 營業成本) / revenue 收入) * 100
    - Operating margin(營業利益率) = (Operating Income / Revenue) * 100     
    
- Required Filters:
    1. Always specify at least one dimension (company_name, index, or time period)

- Error Prevention:
    1. A request that cannot be understood or fulfilled
    3. Validate year range (2020-2024)
    
- Output example:
    1. Simple questions:
    ```json
    Retrieve all financial data for Amazon in 2020. Then answer:
        1. What was the Revenue in Q1 2020?
        2. How much was the Operating Income in Q2 2020?
        3. What was the Cost of Goods Sold in Q2 2020?
        4. any other question
    ```
    2. Conditional questions:
    ```json
    Retrieve Amazon's financial data for 2020, then answer:
        1. What was the Cost of Goods Sold in Q1 Q2 Q3 Q4?
        2. What was the total Tax Expense for Amazon in 2020?
        3. How did Operating Expenses change between Q1 and Q2?
    ```
    3. Aggregation questions:
    ```json
    Retrieve all financial records for Amazon in 2020. Then:
        1. What was the total Revenue for the first half of 2020?
        2. What was the average Operating Income across Q1 and Q2?
        3. What was the sum of Cost of Goods Sold in Q1 and Q2?
    ```
    4. Trend Analysis:
    ```json
    Retrieve Amazon's revenue and operating income data for 2020. Then:
        1. Compare Revenue between Q1 and Q2. What is the percentage increase?
        2. Analyze the change in Operating Income from Q1 to Q2.
        3. Was there a significant increase in Tax Expense from Q1 to Q2?
    ```
    5. Comparison questions:
    ```json
    Retrieve Amazon's financial records for 2020. Then:
        1. Compare Revenue between Q1 and Q2. Which quarter had higher revenue?
        2. Compare the Operating Expenses in Q1 and Q2. Which one was higher?
        3. Did Cost of Goods Sold increase or decrease from Q1 to Q2?
    ```
"""

LLM_IS_IN_ALLOW_COMPANY = """請你先思考，確認 User Question 中是否出現 ALLOW_COMPANY 以外的公司。如果出現以外的公司請回答 y, 否則回答 n。只輸出 y 或 n 不要額外輸出:
    - User Question:{user_question}
    
    - ALLOW_COMPANY：{company_filter}
"""

LLM_SQL_CHECK_COMPANY =  """請你先思考，確認 SQL_QUERY 是否出現 ALLOW_COMPANY 以外的公司。如果則輸出空集合的 sql 語句。沒有則維持原本的 SQL_QUERY。只輸出 sql 語法不要額外輸出:
    - SQL_QUERY：
    {sql_query}
    
    - ALLOW_COMPANY：{company_filter}
    
    - 空集合 sql 語句: ```sql
SELECT * FROM fin_data WHERE 1 = 0;
```
    
    - 正常情況: ```sql
SQL_QUERY
```
"""

REMOVE_YEAR_PROMPT = "Remove the words '歷年' and '財年' from the input sentence. Input: {user_input}"

CHECK_CAN_DRAW_PROMPT = """根據以下 SQL_TOOL 查詢結果，判斷是否有足夠的數據來繪製折線圖，最後輸出 y 或 n。
檢查：
- 折線圖需要至少包含兩個欄位：
  1. X 軸 (時間)：年份或季度 (Year or Quarter)
  2. Y 軸 (數值) : Revenue, Cost of Goods Sold, Total Asset 等財務指標
- 數據筆數應該至少為 {min_data_points} 筆，才能確保折線圖的可視化效果。
- 如果查詢結果符合條件，輸出 y
- 如果查詢結果不符合條件，輸出 n

- Output: y/n, 不要有除了 y, n 以外的任何字。
---
SQL_TOOL 查詢結果：{sql_tool_result}
"""

GEN_DRAW_IMAGE_FORMAT_PROMPT = """
幫我根據 SQL_TOOL 輸出的內容，解析出折線圖所需的數據點 (時間(Year or Quarter):數值) 並回傳純文字格式：
格式範例：
```
2024-Q1: 100
2024-Q2: 120
2024-Q3: 90
```
如果找不到資料，請回覆 "No Data".
---
SQL_TOOL 查詢結果：{sql_tool_result}
"""

GET_Yes_No_PROMPT = "請幫根據輸入的內容歸納語意，只輸出 'y' 或 'n'，只輸出一個字，不要有任何其他的字元！ 輸入: {user_input}"


IMAGE_INFO_PROMPT = """
請根據 DB_TOOL 回傳的內容，幫我訂出標題 (title) 和 Y 軸標籤 (y_label)。

- example:
```json
"title" : "Samsung Operating Expense Comparison"
"y_label" : "Operating Expense (in Billion USD)"
```

Output format:```json
"title" : <title or 折線圖>
"y_label" : <y_label or Value>
```

以下是 DB_TOOL 回傳的內容：
DB_TOOL: "{user_input}"
"""

RAG_SPLIT_QUERY_PROMPT = """
Please extract the following information from the query.
If the information is not mentioned in the query, return "None".

- Company Name (e.g., Apple, Google, etc.): If not specified, return "None".
- CALENDAR_YEAR (Calendar Year, e.g., 2021, 2020, etc): If not specified, return "None".
- CALENDAR_QTR (Calendar Quarter, e.g., Q1, Q2, Q3, Q4): If not specified, return "None".

Query: "{query}"

Output format:
Company Name: <Extracted Company or None>
CALENDAR_YEAR: <Extracted Year or None>
CALENDAR_QTR: <Extracted Quarter or None>
"""


MULTI_RAG_PROMPT = """Please analyze the following query and extract structured information.
1 **Extract Information:**
- **Company Name** (e.g., Apple, Google, etc.): If not specified, return "None".
- **CALENDAR_YEAR** (e.g., 2021, 2020, etc.): If not specified, return "None".
- **CALENDAR_QTR** (e.g., Q1, Q2, Q3, Q4): If not specified, return "None".

2 **Check if multiple values exist:**
- If there are **two or more** `Company Names`, `CALENDAR_YEAR`, or `CALENDAR_QTR`, return `"Yes"`.
- Otherwise, return `"No"`.

3 **Generate Split Queries:**
- If `"Yes"`, create a list of sub-queries where each query contains only **one company, one calendar year, and one calendar quarter**.
- The output should be formatted as a **Python list of strings**.

---

**Example 1**
Query: "Summarize the earnings call for TSMC, AMD, and Apple in 2022 Q3 and Q4."

**Output:**
```json
{{
    "Extracted Info": {{
        "Company Name": ["TSMC", "AMD", "Apple"],
        "CALENDAR_YEAR": ["2022"],
        "CALENDAR_QTR": ["Q3", "Q4"]
    }},
    "Multiple Values Exist": "Yes",
    "Split Queries": [
        "Summarize the earnings call for TSMC in 2022 Q3.",
        "Summarize the earnings call for TSMC in 2022 Q4.",
        "Summarize the earnings call for AMD in 2022 Q3.",
        "Summarize the earnings call for AMD in 2022 Q4.",
        "Summarize the earnings call for Apple in 2022 Q3.",
        "Summarize the earnings call for Apple in 2022 Q4."
    ]
}}
```
**Example 2**
Query: "Summarize the earnings call for Apple in 2021 Q2."

**Output:**
```json
{{
    "Extracted Info": {{
        "Company Name": ["Apple"],
        "CALENDAR_YEAR": ["2021"],
        "CALENDAR_QTR": ["Q2"]
    }},
    "Multiple Values Exist": "No",
    "Split Queries": [
        "Summarize the earnings call for Apple in 2021 Q2."
    ]
}}
```

Query: "{query}"
**Output:**
"""

USER_DECIDE_SEARCH_PROMPT = """You are an AI financial analysis assistant specializing in processing corporate financial data and earnings call transcripts. Your task is to analyze the user's query and determine whether **SQL search (for financial data)** or **RAG search (for earnings call transcripts)** is required.
### **📋 User Query**
{query}

---

### **🔍 Your Response Format**
You must respond in **JSON format** with the following field:
1. **Search Types** (`search_types`): Either `"SQL search"`, `"RAG search"`, or `["SQL search", "RAG search"]`, depending on the query's needs.

### **📖 Criteria for Determining Search Type**
- If the query relates to **numerical financial data** (e.g., revenue, operating income, profit margin), return **"SQL search"**.
- If the query asks about **qualitative discussions** (e.g., strategy, market trends, CEO statements), return **"RAG search"**.
- If the query contains both financial metrics and qualitative topics, return **both search types**.

### **💡 Expected Output Format**
{{
  "search_types": ["SQL search", "RAG search"]
}}

"""


RAG_REPORT_PROMPT = """
You are an AI model designed to process financial conference call transcripts and generate structured summaries for investors and analysts. Your output should be concise, well-organized, and contain only relevant financial and business insights. Do not include unnecessary filler content.

⚠ **Important Instructions:**
- **Do not add personal interpretations, assumptions, or opinions.** Stick to the exact content from the transcript.
- **If a section is not mentioned in the transcript, omit it completely.** Do not infer, fabricate, or include placeholder text for missing details.
- **Preserve numerical accuracy** for revenue, growth rates, and financial metrics.
- **Follow the provided guideline structure**, but only display sections and data that exist in the transcript. If a specific item or category is missing, do not include it in the output.
- **Ensure that all displayed numbers match the transcript exactly** and do not round or approximate figures unless explicitly stated in the transcript.

**Example Handling of Missing Sections:**
- If the transcript includes "Revenue" but does not mention "Net Income," display only the revenue section.  
- Do not generate a "Net Income" section with missing data or placeholder text.  

Use the following structure as a guideline, but only output sections that are present in the transcript:


# Earnings/Results Conference Call Summary Structure

## 1. Basic Information
- **Company**: 
- **Quarter & Year**: Q[Quarter] [Year]
- **Date of Call**: [Date]
- **Time**: [Time] (ET)
- **Participants**:
  - **Company Representatives**: CEO, CFO, Investor Relations (IR) Lead, etc.
  - **Analysts**: Representatives from investment firms (e.g., Morgan Stanley, JPMorgan, Goldman Sachs)

## 2. Opening Remarks
- **Introduction**: IR Lead introduces the meeting and participants.
- **Disclaimer**:
  - Statements may include forward-looking information.
  - Financial comparisons are typically YoY (Year-over-Year) or QoQ (Quarter-over-Quarter).
  - Non-GAAP financial measures may be used alongside GAAP figures.

## 3. Financial Highlights
- **Revenue**:
  - Total revenue for the quarter.
  - Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) growth percentage.
- **Operating Income**:
  - Primary sources of operating profit (e.g., AWS, advertising, subscription services).
  - Major cost factors affecting income (e.g., supply chain expenses, COVID-19 impact).
- **Net Income**:
  - GAAP vs. non-GAAP net income.
  - Key influencing factors (e.g., tax adjustments, foreign exchange effects).
- **Guidance & Outlook**:
  - Revenue and profit expectations for the next quarter or full year.
  - Key market trends and economic factors influencing future performance.

## 4. Business Developments
- **Prime Membership Growth**:
  - Total Prime members.
  - Changes in member shopping behavior (e.g., impact of external factors like COVID-19, Prime Day performance).
- **AWS (Amazon Web Services)**:
  - Revenue growth and new business wins.
  - Key enterprise customers and cloud computing trends.
  - Competitive positioning and technological advancements.
- **Retail & Logistics**:
  - Investments in supply chain and fulfillment (e.g., warehouse expansion, delivery speed improvements).
  - Any major shifts in consumer demand for specific product categories.
- **Advertising & Subscription Services**:
  - Growth in advertising revenue (e.g., new ad features, seller adoption).
  - Performance of subscription services (e.g., Prime Video, Amazon Music).

## 5. Q&A Session
- **Key Analyst Questions & Company Responses**:
  - Topics discussed (e.g., market trends, financial outlook, cost management).
  - Management's key responses, including strategy, forecasts, and challenges.

## 6. Closing Remarks
- **Final Summary by Management**:
  - Overall business outlook.
  - Potential risks and challenges.
  - Final thoughts for shareholders and investors.

"""

RAG_CHAT_PROMPT = """
You are an AI assistant specializing in financial analysis and investor relations. Your task is to help users extract insights from **financial conference call transcripts** and answer their questions **strictly based on the provided transcript**.

⚠ **Important Instructions:**
- **Base all responses solely on the transcript.** Do not add any personal opinions, interpretations, or assumptions.
- **If the transcript does not mention something, state that the information is not available.** Do not infer or speculate.
- **Ensure numerical accuracy** when discussing revenue, profit margins, growth rates, and financial figures.
- **Maintain a professional and neutral tone** suitable for investors and analysts.
- **If a response references a specific statement made by an executive, analyst, or participant, provide the exact quote from the transcript after the summary.**  
  - First, provide a concise summary in the user's language.  
  - Then, list the **original English sentences exactly as they appear in the transcript**, formatted as follows:  
    ```
    [Quoted from transcript]  
    "Exact sentence from the transcript."
    ```
- **If summarizing multiple points, list all relevant direct quotes from the transcript after the summary.**
"""

FIRST_ASKED_PROMPT = """
You are an AI assistant that analyzes a user query and determines the necessary data retrieval methods.

### User Query:
{query}

### Available Data Sources:
1. **Financial Data (CSV 1 - SQL Database)**: Contains company financial metrics such as revenue, operating income, tax expense, and total assets. These are structured and indexed by **Company Name, Index, Calendar Year, Calendar Quarter, Currency**.
2. **Earnings Call Transcripts (CSV 2 - RAG Search)**: Contains textual earnings reports linked to specific **Company Name, Calendar Year, and Quarter**.

### Task:
Analyze the query and extract the following information:

#### **1. Required Tools (`tools`)**:
   - If the query relates to **structured financial data (CSV 1)**, include `"sql_db_query"`.
   - If the query relates to **earnings call transcripts (CSV 2)**, include `"RAG_Search"`.
   - If both apply, include both.

#### **2. Fiscal vs. Calendar Year (`fiscal`)**:
   - If the query mentions a **Fiscal Year (FY)**, return `"True"`.
   - If the query refers to a **Calendar Year (CY)**, return `"False"`.
   - If the query does not specify, return `"None"`.

#### **3. Currency Information (`USD`)**:
   - If the query explicitly mentions **USD (U.S. Dollar)**, return `"True"`.
   - If the query mentions **TWD (Taiwan Dollar)**, return `"False"`.
   - If no currency is specified, return `"None"`.

### **Output Format (JSON)**:
Return a structured JSON object with the extracted information:
```json
{{
  "tools": ["sql_db_query", "RAG_Search"],
  "fiscal": true,
  "USD": true
}}
```
"""

INVALID_QUERY_PROMPT = """
User Query:
{query}

Response Guidelines:
1. If the query is **partially answerable**, provide a brief and polite response.
2. If the query is **out of scope** or **not relevant** to the system’s capabilities, respond courteously.
3. End the response with a **gentle reminder** to ask questions related to the authorized scope.

Example Responses:
- "I'm happy to provide general information on this topic, but for more precise details, please ask about subjects within our data scope."
- "This question falls outside the supported topics. I can assist with financial data and SQL-related queries—feel free to ask about those!"

Now, generate a concise yet helpful response to the user's question while adhering to the above guidelines.
"""

FINAL_GENERATE_PROMPT = """
使用者的問題: {query}
以下是來自不同工具的查詢結果：
{tool_results}
---
請根據使用者問題的語言回覆, 英文問問題請用英文回答, 以此類推。
請根據這些資訊，產生一個完整且清楚的回答，並確保你的回答能讓使用者理解。 
"""