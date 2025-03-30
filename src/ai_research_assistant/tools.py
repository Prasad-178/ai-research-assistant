from langchain_community.tools import ArxivQueryRun, TavilyAnswer
import os
import dotenv
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from composio_langgraph import Action, ComposioToolSet, App
from langchain_community.utilities import ArxivAPIWrapper

composio_toolset = ComposioToolSet()

dotenv.load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


def arxiv_search(query: str) -> any:
  """Search for papers, articles and other resources on Arxiv"""
  arxiv = ArxivAPIWrapper()
  docs = arxiv.run(query)
  
  return docs

# result = arxiv_search("Summarize recent advancements in using Graph Neural Networks for drug discovery")
# print(result)
# print(type(result))
