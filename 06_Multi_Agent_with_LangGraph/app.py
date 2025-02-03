from typing import Dict, Any, Callable, List, Optional, TypedDict, Union, Annotated, Tuple
from typing_extensions import TypedDict
import operator

from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from agents.utils import create_team_supervisor, create_working_directory, print_mermaid_image
from agents.rag_chain import create_rag_chain
from agents.research_team import create_research_team
from agents.writing_team import create_writing_team

gpt4oMini = ChatOpenAI(model="gpt-4o-mini")
gpt4turbo = ChatOpenAI(model="gpt-4-turbo")
gpt4 = ChatOpenAI(model="gpt-4")

WORKING_DIRECTORY = Path(create_working_directory())

rag_chain = create_rag_chain(gpt4oMini)
research_team = create_research_team(gpt4turbo, rag_chain)
writing_team = create_writing_team(gpt4turbo, WORKING_DIRECTORY)

supervisor_node = create_team_supervisor(
    gpt4turbo,
    "You are a supervisor tasked with managing a conversation between the"
    " following teams: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When all workers are finished,"
    " you must respond with FINISH.",
    ["Research team", "LinkedIn team"],
)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

def get_last_message(state: State) -> str:
    return state["messages"][-1].content

def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}

supervisor_graph = StateGraph(State)
supervisor_graph.add_node("Research team", get_last_message | research_team | join_graph)
supervisor_graph.add_node(
    "LinkedIn team", get_last_message | writing_team  | join_graph
)
supervisor_graph.add_node("supervisor", supervisor_node)

supervisor_graph.add_edge("Research team", "supervisor")
supervisor_graph.add_edge("LinkedIn team", "supervisor")
supervisor_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "LinkedIn team": "LinkedIn team",
        "Research team": "Research team",
        "FINISH": END,
    },
)
supervisor_graph.set_entry_point("supervisor")
supervisor = supervisor_graph.compile()

# print_mermaid_image(supervisor_graph, './images/supervisor_graph.png')

#####
for s in supervisor.stream(
    {
        "messages": [
            HumanMessage(
                content="Write a LinkedIn post on the paper 'Extending Llama-3’s Context Ten-Fold Overnight'. First consult the research team. Then make sure you consult the LinkedIn team, and check for copy editing and dopeness, and write the file to disk."
            )
        ],
    },
    {"recursion_limit": 30},
):
    if "__end__" not in s:
        print(s)
        print("---")
#####

# 🏗️ Activity #1 (Bonus Marks)
# 
# Allow the system to dynamically fetch Arxiv papers instead of hard coding them.
# 
# > HINT: Tuesday's assignment will be very useful here.
# 

# ❓ Question #1:
# Why is a "powerful" LLM important for this use-case?
#
# A: It must consider all the context of the request in order to provide the most accurate information possible.
# 
# What tasks must our Agent perform that make it such that the LLM's reasoning capability is a potential limiter?
#
# A: Reasoning, the Agent must be able to understand the order of operations that must be performed in order to complete the request.
# 
# 🏗️ Activity #2:
# 
# Using whatever drawing application you wish - please label the flow above on a diagram of your graph.
# 

# ❓ Question #2:
# 
# How could you make sure your Agent uses specific tools that you wish it to use? Are there any ways to concretely set a flow through tools?
# 
# A: By creating a tool that is specifically designed to be used in conjunction with the Agent, you can ensure that the Agent uses the tool as intended. 
# You can also set a flow through tools by creating a specific path for the Agent to follow that includes the use of the tool.
# 
