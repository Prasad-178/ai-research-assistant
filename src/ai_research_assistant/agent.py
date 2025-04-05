from langchain_openai import ChatOpenAI
import os
from tools import arxiv_search, firecrawl_search, multiply, add, subtract, divide
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
import uuid

def create_agent() -> tuple[CompiledStateGraph, RunnableConfig]:
  llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
  tools = [arxiv_search, firecrawl_search, multiply, add, subtract, divide]

  system_message = SystemMessage(
    content="You are a helpful research assistant tasked with summarizing papers and articles and helping with research."
  )

  memory = MemorySaver()
  react_agent = create_react_agent(model=llm, tools=tools, checkpointer=memory, prompt=system_message)

  config = RunnableConfig(
    run_name="research_assistant",
    run_id=str(uuid.uuid4()),
    configurable={
      "thread_id": "1"
    }
  )

  return react_agent, config

def invoke(agent: CompiledStateGraph, config: RunnableConfig, message: str):
  return agent.stream({"messages": [HumanMessage(content=message)]}, config, stream_mode="values")

# agent, config = create_agent()
# result = agent.invoke({"messages": [HumanMessage(content='Summarize recent advancements in using Graph Neural Networks for drug discovery')]}, config)

