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


