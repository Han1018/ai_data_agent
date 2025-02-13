import streamlit as st
from streamlit_chat import message
from langchain.chat_models import init_chat_model
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from sql_search import get_sql_tools
from rag_search import get_rag_tools
from config import MODEL_NAME, MODEL_PROVIDER

DB_FILE = 'user_db.json'
if not os.path.exists(DB_FILE):
    with open(DB_FILE, 'w') as file:
        db = {"users": {}}
        json.dump(db, file)
else:
    with open(DB_FILE, 'r') as file:
        db = json.load(file)

def save_db():
    with open(DB_FILE, 'w') as file:
        json.dump(db, file)

# Loading the model of your choice
llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
sql_tools = get_sql_tools()
rag_tools = get_rag_tools()
agent = Agent(model=llm, sql_tools=sql_tools, rag_tools=rag_tools)

def conversational_chat(query):
            history = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state['history']])
            full_query = f"{history}\nUser: {query}\nBot:"
            
            response = agent.run(full_query).content  # 用 agent 來處理查詢
            # **替換 ⏳ ... 為 AI 回應**
            st.session_state['history'][-1] = (query, response)
            return response

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
    st.sidebar.write(f"Current User Role: {user_role}")
    st.sidebar.write(f"Current Page: {page}")



    if page == "💬 Chat Mode":
        st.subheader("💬 AI ChatBot query")
        # Function for handling conversation with history

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
        user_input = st.chat_input(f"Start chatting as {user_role}...")

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
            "🇰🇷 Korea Data Viewer": ["Samsung"],
            "🇨🇳 China Data Viewer": ["Baidu", "Tencent"],
            "🌍 Global Data Viewer": ["Amazon","AMD","Amkor","Apple","Applied Material","Baidu","Broadcom","Cirrus Logic","Google","Himax","Intel","KLA","Marvell","Microchip","Microsoft","Nvidia","ON Semi","Qorvo","Qualcomm","Samsung","STM","Tencent","Texas Instruments","TSMC","Western Digital"]
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
        st.success(f"Welcome back, {st.session_state['username']}! Role: {st.session_state['user_role']}")
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
        if username in db["users"] and db["users"][username]["password"] == password:
            role = db["users"][username]["role"]  # 讀取角色
            st.success(f"Welcome, {username}! Role: {role}")
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
        "cn123": "🇨🇳 China Data Viewer",
        "kr123": "🇰🇷 Korea Data Viewer",
        "g123": "🌍 Global Data Viewer"
    }

    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col3:
        if st.button("Submit", use_container_width=True):
            if username in db["users"]:
                st.error("Username already exists. Try another one.")
            elif access_token not in valid_tokens:
                st.error("Invalid access token.")
            else:
                # 註冊並存入使用者角色
                db["users"][username] = {"password": password, "role": valid_tokens[access_token]}
                save_db()
                st.success("Account created successfully! Redirecting to login...")

                # 回到登入畫面
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
    🌍 Global Data Viewer: `g123`"""
)



if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state.get("logged_in", False):
        main()
    else:
        login_or_signup()