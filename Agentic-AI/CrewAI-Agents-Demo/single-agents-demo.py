import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

search_tool = SERPER_API_KEY()
llm = ChatOpenAI(model="gpt-3.5-turbo")

def create_research_agent():
    return Agent(
            role = "Research Specialist",
            goal = "Conduct through research on topics",
            backstory = "You are an experienced researcher with expertise in finding and synthesizing information from variance source",
            verbose = True,
            allow_delegation = False,
            tools = [search_tool],
            llm = llm
    )

def create_research_task(agent, task):
    return Task(
            description = f"Research the following topic and provide a commprehensive summary: {topic}",
            agent = agent,
            excepted_output = "A detailed summary of the research findings, including key points and insights related to an topics"
    )

def run_research(topic):
    agent = create_research_agent()
    task = create_research_task()
    crew = crew(agent = [agent], task =[task])
    results = crew.kickoff()
    return results

if __name__ == "__main__":
    print("Welcome to the Research Agents")
    topic = input ("Enter the research topic:")
    result = run_research(topic)
    print("Research Results:")
    print(result)