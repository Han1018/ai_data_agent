LLM_SQL_SYS_PROMPT = """請將以下查詢轉換成具體且明確的 user query: {query}
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