import os
import dotenv
from typing import Union, List
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool
from firecrawl import FirecrawlApp

dotenv.load_dotenv()
if not os.environ.get("TAVILY_API_KEY"):
  os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
if not os.environ.get("FIRECRAWL_API_KEY"):
  os.environ["FIRECRAWL_API_KEY"] = os.getenv("FIRECRAWL_API_KEY")

app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

@tool
def multiply(a: float, b: float) -> float:
  """Multiply two numbers"""
  return a * b

@tool
def add(a: float, b: float) -> float:
  """Add two numbers"""
  return a + b

@tool
def subtract(a: float, b: float) -> float:
  """Subtract two numbers"""
  return a - b

@tool
def divide(a: float, b: float) -> float:
  """Divide two numbers"""
  return a / b

@tool
def arxiv_search(query: str) -> str:
  """Search for papers, articles and other resources on Arxiv"""
  arxiv = ArxivAPIWrapper()
  docs = arxiv.run(query)

  return docs

@tool
def firecrawl_search(url: Union[str, List[str]]) -> str:
  """Extract and index information from a link or a list of links"""
  if isinstance(url, str):
    urls = [url]
  else:
    urls = url

  overall_content = []
  for url in urls:
    scrape_status = app.scrape_url(
      url=url,
      params={
        'formats': ['markdown']
      }
    )
    overall_content.append(scrape_status['markdown'])

  return "\n\n".join(overall_content)

# result = arxiv_search("Summarize recent advancements in using Graph Neural Networks for drug discovery")
# print(result)
# print(type(result))
