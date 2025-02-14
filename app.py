import streamlit as st
from streamlit_chat import message
from langchain.chat_models import init_chat_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import agent


import psycopg2
from sqlalchemy import create_engine
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
engine = create_engine(db_url)

conn = psycopg2.connect(
    dbname=DATABASE,
    user=_USER,
    password=_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

import bcrypt

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def save_user(username, password, role):
    """儲存使用者到 PostgreSQL"""
    hashed_pw = hash_password(password)  # 加密密碼
    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", 
            (username, hashed_pw, role)
        )
        conn.commit()
    except psycopg2.IntegrityError:
        conn.rollback()
        st.error("Username already exists. Try another one.")

def authenticate_user(username, password):
    """從 SQL 驗證使用者"""
    cursor.execute("SELECT password, role FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    
    if result:
        hashed_pw, role = result
        if verify_password(password, hashed_pw):
            # **登入成功後存入 Session**
            st.session_state["username"] = username
            st.session_state["user_role"] = role
            return role  # 登入成功，回傳使用者角色
    return None  # 登入失敗


USERROLE = {"KR": "🇰🇷 Korea Data Viewer", "CN": "🇨🇳 China Data Viewer", "GB": "🌍 Global Data Viewer"}

# Loading the model of your choice
llm = init_chat_model("gemini-1.5-pro", model_provider="google_vertexai")


def main():
    st.title("📊 App")

    # Sidebar for selecting user role
    user_role = st.session_state.get("user_role", "🇰🇷 Korea Data Viewer")  # 若未設置則給予預設值
    username = st.session_state.get("username", "Guest")

    st.sidebar.title(f"👋 Welcome! **{username}**")
    page = st.sidebar.radio("Select operating mode", ["💬 Chat Mode", "📈 Report Mode"])

    # Reset session state when role changes
    if 'previous_role' not in st.session_state or st.session_state['previous_role'] != user_role:
        temp_logged_in = st.session_state.get("logged_in", False)
        temp_username = st.session_state.get("username", "")
        st.session_state.clear()  # 清除 session，但登入狀態要還原
        st.session_state['previous_role'] = user_role
        st.session_state["logged_in"] = temp_logged_in
        st.session_state["username"] = temp_username
        st.session_state["user_role"] = user_role

    # Display the selected user role in the sidebar
    st.sidebar.write(f"Current User Role: {USERROLE[user_role]}")
    st.sidebar.write(f"Current Page: {page}")



    if page == "💬 Chat Mode":
        st.subheader("💬 AI ChatBot query")
        # Function for handling conversation with history
        def conversational_chat(query):
            history = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state['history']])
            full_query = f"{history}\nUser: {query}\nBot:"
            
            response = agent.run(full_query).content  # 用 agent 來處理查詢
            # **替換 ⏳ ... 為 AI 回應**
            st.session_state['history'][-1] = (query, response)
            return response

        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        message("Hello! How can I assist you today?", avatar_style="thumbs")

        # Custom CSS to make the chat input box fixed at the bottom
        st.markdown(
            """
            <style>
            /* Try applying to more general Streamlit chat input elements */
            .stTextInput, .stChatInput, .stTextArea { 
                position: fixed; 
                bottom: 0; 
                z-index: 9999;
            }
            </style>
            """, unsafe_allow_html=True
        )

        # Chat input using st.chat_input
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state['history']):
                message(user_msg, is_user=True, key=f"user_{i}")
                message(bot_msg, key=f"bot_{i}", avatar_style="thumbs")

        # **聊天輸入框**
        user_input = st.chat_input(f"Start chatting as {USERROLE[user_role]}...")

        if user_input:
        # **立即顯示 User Input 並先加上 ⏳ ...**
            with chat_container:
                message(user_input, is_user=True, key=f"user_{len(st.session_state['history'])}_new")
                message("⏳ ...", key=f"bot_{len(st.session_state['history'])}_new", avatar_style="thumbs")

            # **先存入 ⏳ ...，待 AI 回應後替換**
            st.session_state['history'].append((user_input, "⏳ ..."))

            # **異步處理 AI 回應**
            bot_response = conversational_chat(user_input)  

            # **更新 ⏳ ... 為 AI 回應**
            st.session_state['history'][-1] = (user_input, bot_response)  # 取代最後一筆 "⏳ ..."

            # **刷新 UI 讓變更生效**
            st.rerun()

        # **滾動到底部標記**
        st.markdown("<div id='scroll-bottom'></div>", unsafe_allow_html=True)

        # **使用 JavaScript 自動滾動到底部**
        st.markdown(
            """
            <script>
            var scrollBottom = document.getElementById("scroll-bottom");
            if (scrollBottom) {
                scrollBottom.scrollIntoView({ behavior: "smooth" });
            }
            </script>
            """, unsafe_allow_html=True
        )
    elif page == "📈 Report Mode":
        st.subheader("📈 Summarized report")

        # **available companies based on user role**
        company_options = {
            "KR": ["Samsung"],
            "CN": ["Baidu", "Tencent"],
            "GB": ["Amazon","AMD","Amkor","Apple","Applied Material","Baidu","Broadcom","Cirrus Logic","Google","Himax","Intel","KLA","Marvell","Microchip","Microsoft","Nvidia","ON Semi","Qorvo","Qualcomm","Samsung","STM","Tencent","Texas Instruments","TSMC","Western Digital"]
        }
        available_companies = company_options[user_role]

        # **Select company**
        company = st.selectbox("Select company", available_companies)

        # **Select quarter**
        quarter = st.selectbox("Select quarter", ["Q1", "Q2", "Q3", "Q4"])

        # **data**
        np.random.seed(42)
        data = {
            "Month": ["Jan", "Feb", "Mar"] if quarter == "Q1" else
                    ["Apr", "May", "Jun"] if quarter == "Q2" else
                    ["Jul", "Aug", "Sep"] if quarter == "Q3" else
                    ["Oct", "Nov", "Dec"],
            "Revenue ($M)": np.random.randint(50, 200, size=3),
            "Profit ($M)": np.random.randint(10, 100, size=3)
        }

        df = pd.DataFrame(data)

        st.write(f"### 📌 {company} - {quarter} 財務數據")
        
        st.dataframe(df)

        # **plot**
        st.subheader(f"📈 {company} - {quarter} 收入與利潤趨勢")
        fig, ax = plt.subplots()
        df.set_index("Month").plot(ax=ax, marker='o')
        st.pyplot(fig)

def login_or_signup():
    st.title("🔑 Login or Create Account")

    # 切換登入或註冊
    if "signup_mode" not in st.session_state:
        st.session_state["signup_mode"] = False

    if st.session_state["signup_mode"]:
        signup_page()  # 顯示註冊頁面
        return

    # 登入頁面
    if st.session_state.get("logged_in", False):
        st.success(f"Welcome back, {st.session_state['username']}! Role: {USERROLE[st.session_state['user_role']]}")
        return

    username = st.text_input("Username", value=st.session_state.get("username", ""))
    password = st.text_input("Password", type="password")
    
    col1, col2, col3 = st.columns([1, 3, 1])  # 左中右三欄

    with col3:
        login = st.button("Login", use_container_width=True)

    with col1:
        if st.button("Create Account"):
            st.session_state["signup_mode"] = True  # 切換到註冊頁面
            st.rerun()  # 重新載入頁面

    if login:
        role = authenticate_user(username, password)  # 查詢 SQL
        if role:
            st.success(f"Welcome, {username}! Role: {USERROLE[role]}")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_role"] = role
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

def signup_page():
    st.write("### 📝 Create an Account")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    access_token = st.text_input("Access Token", type="password")

    valid_tokens = {
        "cn123": "CN",
        "kr123": "KR",
        "gb123": "GB"
    }

    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col3:
        if st.button("Submit", use_container_width=True):
            if access_token not in valid_tokens:
                st.error("Invalid access token.")
            else:
                save_user(username, password, valid_tokens[access_token])  # 存入 SQL
                st.success("Account created successfully! Redirecting to login...")
                st.session_state["signup_mode"] = False
                st.rerun()

    with col1:
        if st.button("↩️"):
            st.session_state["signup_mode"] = False  # 切換回登入模式
            st.rerun()
    st.write("""
    ##### Access Tokens
    🇨🇳 China Data Viewer: `cn123`\n
    🇰🇷 Korea Data Viewer: `kr123`\n
    🌍 Global Data Viewer: `gb123`"""
)



if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state.get("logged_in", False):
        main()
    else:
        login_or_signup()