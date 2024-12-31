from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo # Used for Web searched
import openai

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

## Web search agent
web_search_agent=Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tool_calls=True,
    markdown=True,
)

## Financial agent
finance_agent=Agent(
    name="Web search Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)
    ],
    instructions=["Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    model=Groq(id="Llama-3.2-3b-Preview"),
    team=[web_search_agent, finance_agent],
    instructions=["Alway include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("summarize analyst recommendation and share the latest news for NVDA", stream=True)