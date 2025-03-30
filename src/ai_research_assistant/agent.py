from langchain_openai import ChatOpenAI
import os
from tools import arxiv_search
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
tools = [arxiv_search]

llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(
  content="You are a helpful research assistant tasked with summarizing papers and articles and helping with research."
)

def assistant(state: MessagesState):
  return {
    "messages": [
      llm_with_tools.invoke([system_message] + state["messages"])
    ]
  }

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
graph = builder.compile()

print(graph._draw_graph)
