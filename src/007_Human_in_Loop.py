from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel
from pydantic import Field

load_dotenv("../.env.gpt")

class EndpointConfig(BaseSettings):
    hf_token: str 
    repo_id: str 
    provider: str 
    temperature: float 

config = EndpointConfig()

class State(BaseModel):
    messages: Annotated[list, Field(default_factory=list), add_messages]

@tool
def get_stock_price(symbol: str) -> float:
    """
    Return the current price of a stock given the stock symbol
    """

    stock_prices = {
        "RELIANCE": 1359.10,
        "HDFCBANK": 960.95,
        "AIRTELPP": 1446.55,
        "BHARTIARTL": 1879.80,
        "TCS": 3097.80,
        "ICICIBANK": 1408.00,
        "SBIN": 809.85,
        "HINDUNILVR": 2665.00,
        "INFY": 1461.50,
        "BAJFINANCE": 933.05,
        "LICI": 880.00,
        "ITC": 415.75,
        "LT": 3592.60,
        "MARUTI": 14680.0,
        "HCLTECH": 1444.20
    }
    return stock_prices.get(symbol.upper(), 0.0)

@tool
def list_all_stocks() -> list:
    """Here is list of all available stocks."""
    return ["RELIANCE", "HDFCBANK", "AIRTELPP", "BHARTIARTL", "TCS", "ICICIBANK", "SBIN", "HINDUNILVR", "INFY", "BAJFINANCE", "LICI", "ITC", "LT", "MARUTI", "HCLTECH"]

@tool
def buy_stocks(symbol: str, quantity: int) -> str:
    """
    Buy a given quantity of stock for the given stock symbol
    """
    price = get_stock_price(symbol)
    total_price = price * quantity
    decision = interrupt(f"Do you want to proceed with the purchase of {quantity} shares of {symbol} for a total price of ₹{total_price:.2f}? (yes/no)")
    if decision == "yes":
        return f"Bought {quantity} shares of {symbol} for ₹{total_price:.2f}"
    else:
        return "Purchase cancelled."
    

tools = [get_stock_price, buy_stocks, list_all_stocks]

class HuggingFaceChatModel(ChatHuggingFace):
    def __init__(self, config: EndpointConfig):
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=config.hf_token,
            repo_id=config.repo_id,
            provider=config.provider,
            temperature=config.temperature
        )
        super().__init__(llm=llm)

chat_model_with_tools = HuggingFaceChatModel(EndpointConfig()).bind_tools(tools=tools)

def chatbot(state: State):
    """
    Chatbot node that uses a chat model with tools to respond to user messages.
    """
    return {"messages": [chat_model_with_tools.invoke(state.messages)]}

memory_saver = MemorySaver()
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory_saver)

config = {"configurable": {"thread_id": "buy_thread"}}

state = graph.invoke({"messages": ["Hello, I want to buy stocks. List available stocks. And I want to buy 10 shares of RELIANCE. Also 10 shares of LT"]}, config=config)
print(state["messages"][-1])

print(state.get("__interrupt__"))

decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print("------------------------------")
print(state["messages"])


