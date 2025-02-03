from typing import Dict, Any, Callable, List, Optional, TypedDict, Union, Annotated, Tuple
from typing_extensions import TypedDict
import functools
import operator

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph

from agents.utils import agent_node, create_agent, create_team_supervisor

tavily_tool = TavilySearchResults(max_results=5)

class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

def enter_research_graph(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results

def create_research_team(llm, rag_chain):    
    @tool
    def retrieve_information(
        query: Annotated[str, "query to ask the retrieve information tool"]
        ):
        """Use Retrieval Augmented Generation to retrieve information about the 'Extending Llama-3’s Context Ten-Fold Overnight' paper."""
        return rag_chain.invoke({"question" : query})

    # search_agent
    search_agent = create_agent(
        llm,
        [tavily_tool],
        "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")

    # research_agent
    research_agent = create_agent(
        llm,
        [retrieve_information],
        "You are a research assistant who can provide specific information on the provided paper: 'Extending Llama-3’s Context Ten-Fold Overnight'. You must only respond with information about the paper related to the request.",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="PaperInformationRetriever")

    #supervisor_agent
    supervisor_agent = create_team_supervisor(
        llm,
        ("You are a supervisor tasked with managing a conversation between the"
        " following workers:  Search, PaperInformationRetriever. Given the following user request,"
        " determine the subject to be researched and respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. "
        " You should never ask your team to do anything beyond research. They are not required to write content or posts."
        " You should only pass tasks to workers that are specifically research focused."
        " When finished, respond with FINISH."),
        ["Search", "PaperInformationRetriever"],
    )

    _research_graph = StateGraph(ResearchTeamState)

    _research_graph.add_node("Search", search_node)
    _research_graph.add_node("PaperInformationRetriever", research_node)
    _research_graph.add_node("supervisor", supervisor_agent)

    _research_graph.add_edge("Search", "supervisor")
    _research_graph.add_edge("PaperInformationRetriever", "supervisor")
    _research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"Search": "Search", "PaperInformationRetriever": "PaperInformationRetriever", "FINISH": END},
    )

    _research_graph.set_entry_point("supervisor")
    research_graph = _research_graph.compile()

    #####
    # Print Mermaid Image
    # try:
    #     image = research_graph.get_graph(xray=True).draw_mermaid_png()
    #     with open('./images/research_graph.png', 'wb') as f:
    #         f.write(image)
    # except Exception as e:
    #     print('Error', e)
    #     pass
    #####

    research_graph_chain = enter_research_graph | research_graph

    return research_graph_chain

    #####
    # for s in research_graph.stream(
    #     "What are the main takeaways from the paper `Extending Llama-3's Context Ten-Fold Overnight'? Please use Search and PaperInformationRetriever!", {"recursion_limit": 100}
    # ):
    #     if "__end__" not in s:
    #         print(s)
    #         print("---")
    #####
