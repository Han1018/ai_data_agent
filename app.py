import streamlit as st
from streamlit_chat import message
from langchain.chat_models import init_chat_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agent import create_agent, AgentState
from config import PROJECT_ID,REGION,BUCKET,BUCKET_URI,INDEX_ID,ENDPOINT_ID,DB_HOST,DB_PORT,DATABASE,_USER,_PASSWORD,MODEL_NAME,MODEL_PROVIDER, EMBEDDING_MODEL_NAME
import psycopg2
from sqlalchemy import create_engine
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
MODE = {"💬 Chat Mode":"Chat Mode", "📈 Report Mode":"Report Mode"}
# # Loading the model of your choice
# llm = init_chat_model("gemini-1.5-pro", model_provider="google_vertexai")


# 初始化 session_state 變數
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState(
        query="",
        adjusted_query="",
        tools=[],
        tool_results=[],
        final_answer="",
        is_first= True,
        is_fiscal= None,
        is_usd= None,
        is_end= False
    )


def main():
    st.title("📊 App")

    # Sidebar for selecting user role
    user_role = st.session_state.get("user_role", "🇰🇷 Korea Data Viewer")  # 若未設置則給予預設值
    username = st.session_state.get("username", "Guest")
    

    
    st.sidebar.title(f"👋 Welcome! **{username}**")
    page = st.sidebar.radio("Select operating mode", ["💬 Chat Mode", "📈 Report Mode"])

    
    # Display the selected user role in the sidebar
    st.sidebar.write(f"Current User Role: {USERROLE[user_role]}")
    st.sidebar.write(f"Current Page: {page}")

    if page == "💬 Chat Mode":
        st.subheader("💬 AI ChatBot query")
        st.session_state["mode"] = MODE[page]  # 確保 session_state 更新
        mode = st.session_state.get("mode", "Chat Mode")  # 預設為 Chat Mode
        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        if 'waiting_for_response' not in st.session_state:
            st.session_state['waiting_for_response'] = None  # 存放等待 AI 回應的訊息  
        # 初始化 Agent
        agent = create_agent(role=user_role, mode=mode)

        message("Hello! How can I assist you today?", avatar_style="thumbs")


        # **顯示歷史對話**
        chat_container = st.container()
        with chat_container:
            for i, entry in enumerate(st.session_state['history']):
                if entry["role"] == "user" and entry["type"] == "text":
                    message(entry["content"], is_user=True, key=f"user_{i}")
                elif entry["role"] == "bot" and entry["type"] == "text":
                    message(entry["content"], key=f"bot_{i}", avatar_style="thumbs")
                elif entry["role"] == "bot" and entry["type"] == "image":
                    img_html = f'<img src="{entry["content"]}" width="250"/>'
                    message(img_html, key=f"img_{i}", allow_html=True, avatar_style="thumbs")  # **顯示圖片**


        # **處理等待中的 AI 回應**
        if st.session_state['waiting_for_response']:
            user_input = st.session_state['waiting_for_response']
            # 先更新 query
            st.session_state.agent_state["query"] = user_input
            # **執行 agent**
            final_answer, end_state = agent.run(user_input, st.session_state.agent_state)
            # **更新 `AgentState`**
            st.session_state.agent_state.update(end_state)  # 直接用 `end_state` 覆蓋原本的 state
            st.session_state.agent_state["final_answer"] = final_answer  # 確保 `final_answer` 也更新
            # **找到最後一筆 "⏳ ..." 並更新**
            for i in range(len(st.session_state['history']) - 1, -1, -1):
                if st.session_state['history'][i]["content"] == "⏳ ...":
                    
                    st.session_state['history'][i] = {"role": "bot", "type": "text", "content": final_answer}  # **直接替換 bot 的回應**
                    # st.session_state['history'].append({"role": "bot", "type": "image", "content": img_url})  # **加入圖片**
                    st.session_state['waiting_for_response'] = None  # 清除等待狀態
                    st.rerun()  # 🔄 重新渲染頁面，讓 AI 回應顯示
                    break
        # **聊天輸入框**
        user_input = st.chat_input(f"Start chatting as {USERROLE[user_role]}...")

        if user_input and st.session_state['waiting_for_response'] is None:  # 只有在沒有等待中的回應時才加入新訊息
                st.session_state['history'].append({"role": "user", "type": "text", "content": user_input})  # 顯示使用者輸入
                st.session_state['history'].append({"role": "bot", "type": "text", "content": "⏳ ..."})  # 顯示等待中的訊息
                st.session_state['waiting_for_response'] = user_input  # 標記等待 AI 回應
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
        if st.button("Login", use_container_width=True):
            role = authenticate_user(username, password)
            if role:
                st.success(f"Welcome, {username}! Role: {USERROLE[role]}")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["user_role"] = role
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    with col1:
        if st.button("Create Account"):
            st.session_state["signup_mode"] = True
            st.rerun()

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
    if st.session_state["logged_in"]:
        main()
    else:
        login_or_signup()