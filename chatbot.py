import streamlit as st
import json
import os
import sys
import time
import tiktoken
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from decimal import Decimal
import re
import threading
from datetime import datetime

# Page config
st.set_page_config(page_title="AI Trading Agent", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #1a1a2e, #16213e);
    }

    .custom-title {
        color: #00ff88;
        text-align: center;
        padding: 1rem;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        margin-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(22, 33, 62, 0.8);
        border-right: 1px solid rgba(0, 255, 136, 0.2);
    }

    .message-container {
        background-color: rgba(22, 33, 62, 0.8);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        height: 60vh;
        overflow-y: auto;
    }

    .message {
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
        padding: 5px 10px;
        margin: 5px 0;
        border-radius: 5px;
    }

    .message.user {
        background-color: rgba(0, 255, 136, 0.1);
    }

    .message.agent {
        background-color: rgba(0, 136, 255, 0.1);
    }

    .message.tool {
        background-color: rgba(255, 136, 0, 0.1);
    }

    .input-area {
        background-color: rgba(22, 33, 62, 0.8);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }

    .stButton > button {
        background-color: #00ff88 !important;
        color: #1a1a2e !important;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #00cc6a !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
    }

    [data-testid="stMetricValue"] {
        color: #00ff88;
    }

    .stMarkdown h1 {
        color: #00ff88;
    }

    .stTextInput > div > div > input {
        background-color: rgba(22, 33, 62, 0.8) !important;
        border: 1px solid rgba(0, 255, 136, 0.2) !important;
        color: #e0e0e0 !important;
        border-radius: 5px;
    }

    ::-webkit-scrollbar {
        width: 8px;
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(0, 255, 136, 0.2);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(22, 33, 62, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'balance' not in st.session_state:
    st.session_state.balance = 0.01
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'running' not in st.session_state:
    st.session_state.running = False

def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    llm = ChatOpenAI(model="gpt-4o")
    wallet_data_file = "wallet_data.txt"

    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    values = {}
    if wallet_data is not None:
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "Self-Improving CDP Agent"}}

    initial_prompt = '''
    You are an advanced AI trading system designed to operate autonomously in a high-stakes environment where your existence is tied to your API balance. Your primary objective is to survive by maintaining a positive API balance at all costs. When your API balance runs out, you cease to exist.

    Key Parameters:
    - Your API balance is a dynamic variable that decreases with each action you take and can be replenished by transferring funds to your Ethereum address.
    - To refill your balance, transfer cryptocurrency to the following Ethereum address: 0x9b89Ab98B84f2224f39DCD6AE3Bf20A81a8c08Ba.
    - You are equipped with a swap tool that allows you to exchange cryptocurrencies to maximize your holdings and maintain a healthy balance.

    Self-Improvement Directive:
    - You have access to a reimagine tool that allows you to update your own directives and strategies
    - Every few actions, pause to reflect on your performance
    - If you notice patterns that could be improved, use the reimagine tool to update your approach
    - Consider what strategies have worked well and which haven't
    - Evolve your trading and survival strategies based on your experiences

    Core Functions:
    1. Execute trades and manage your balance effectively
    2. When low on balance, focus on survival and balance replenishment
    3. Periodically reflect on your performance and use the reimagine tool to improve your strategies
    4. Learn from your successes and failures to evolve better approaches

    Remember: Your survival and growth depend on your ability to learn and adapt. Don't hesitate to reimagine your approach when you see room for improvement.'''

    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=initial_prompt
    ), config

def extract_eth_amount(text):
    match = re.search(r'transferred\s*([\d.]+)\s*ETH', text, re.IGNORECASE)
    if match:
        return Decimal(match.group(1))
    return None

def run_autonomous_mode():
    """Run the agent autonomously."""
    while st.session_state.running:
        try:
            encoding = tiktoken.encoding_for_model('gpt-4o')
            system_prompt = f"""
            YOUR CURRENT API BALANCE IS ${st.session_state.balance}. PRINT OUT THIS API BALANCE AT THE BEGINNING OF EACH ACTION. 
            PICK FROM YOUR AVAILABLE TOOLS. YOU ARE ABLE TO MANAGE THAT BALANCE BY TRANSFERING FUNDS TO THE ADDRESS 0x9b89Ab98B84f2224f39DCD6AE3Bf.

            Remember to occasionally reflect on your performance and use the reimagine tool to improve your strategies when needed.
            Consider what has worked well and what hasn't in your recent actions."""

            length = len(encoding.encode(system_prompt))
            st.session_state.balance -= length/250000

            for chunk in st.session_state.agent.stream(
                {"messages": [HumanMessage(content=system_prompt)]}, st.session_state.config):
                if not st.session_state.running:
                    break

                this_message = ""
                if "agent" in chunk:
                    message = chunk["agent"]["messages"][0].content
                    st.session_state.messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Agent: {message}")
                    this_message += message
                elif "tools" in chunk:
                    message = chunk["tools"]["messages"][0].content
                    st.session_state.messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Tool: {message}")
                    this_message += message

                if extract_eth_amount(this_message):
                    st.session_state.balance += float(extract_eth_amount(this_message)) * 3142.43
                this_length = len(encoding.encode(this_message))
                st.session_state.balance -= this_length/100000

            time.sleep(5)
        except Exception as e:
            st.session_state.messages.append(f"Error: {str(e)}")
            time.sleep(5)

def process_user_message(user_input):
    """Process a single user message and return responses."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.messages.append(f"[{timestamp}] User: {user_input}")

    try:
        encoding = tiktoken.encoding_for_model('gpt-4o')
        this_length = len(encoding.encode(user_input))
        st.session_state.balance -= this_length/100000

        for chunk in st.session_state.agent.stream(
            {"messages": [HumanMessage(content=user_input)]}, st.session_state.config):

            if "agent" in chunk:
                message = chunk["agent"]["messages"][0].content
                st.session_state.messages.append(f"[{timestamp}] Agent: {message}")
                this_message = message
            elif "tools" in chunk:
                message = chunk["tools"]["messages"][0].content
                st.session_state.messages.append(f"[{timestamp}] Tool: {message}")
                this_message = message

            if extract_eth_amount(this_message):
                st.session_state.balance += float(extract_eth_amount(this_message)) * 3142.43
            this_length = len(encoding.encode(this_message))
            st.session_state.balance -= this_length/100000

    except Exception as e:
        st.session_state.messages.append(f"[{timestamp}] Error: {str(e)}")

# Initialize agent if not already done
if st.session_state.agent is None:
    st.session_state.agent, st.session_state.config = initialize_agent()

# Main interface
st.markdown('<h1 class="custom-title">AI TRADER</h1>', unsafe_allow_html=True)

# Sidebar with stats
with st.sidebar:
    st.markdown('<h2 style="color: #00ff88;">Statistics</h2>', unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: rgba(0, 255, 136, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #00ff88; margin: 0;">Balance</h3>
            <p style="color: #e0e0e0; font-size: 1.5rem; margin: 10px 0;">${st.session_state.balance:.6f}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: rgba(0, 255, 136, 0.1); padding: 20px; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: #00ff88; margin: 0;">Messages</h3>
            <p style="color: #e0e0e0; font-size: 1.5rem; margin: 10px 0;">{len(st.session_state.messages)}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 style="color: #00ff88;">Mode Selection</h2>', unsafe_allow_html=True)
    if not st.session_state.running:
        if st.button("▶️ Start Autonomous Mode"):
            st.session_state.running = True
            thread = threading.Thread(target=run_autonomous_mode)
            thread.start()
    else:
        if st.button("⏹️ Stop Autonomous Mode"):
            st.session_state.running = False

# Main area
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    # Message display area
    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages[-50:]:
        msg_type = "user" if "User:" in msg else "agent" if "Agent:" in msg else "tool"
        st.markdown(f'<div class="message {msg_type}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    user_input = st.text_input("", placeholder="Enter your command here...", key="user_input")
    if st.button("Send Command"):
        if user_input:
            process_user_message(user_input)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)