from typing import Dict, Any, Callable, List, Optional, TypedDict, Union, Annotated, Tuple
from typing_extensions import TypedDict
import functools
import operator

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from agents.utils import agent_node, create_agent, create_team_supervisor, print_mermaid_image
from agents.rag_chain import create_rag_chain

class FactCheckingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str

def enter_fact_checking_graph(message: str, team_members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(team_members),
    }
    return results

def create_fact_checking_team(llm: ChatOpenAI, working_directory) -> str:
    rag_chain = create_rag_chain(llm)

    @tool
    def retrieve_information(
        query: Annotated[str, "query to ask the retrieve information tool"]
    ):
        """Use Retrieval Augmented Generation to retrieve information about the 'Extending Llama-3’s Context Ten-Fold Overnight' paper."""
        return rag_chain.invoke({"question" : query})

    @tool
    def read_document(
        file_name: Annotated[str, "File path to save the document."],
        start: Annotated[Optional[int], "The start line. Default is 0"] = None,
        end: Annotated[Optional[int], "The end line. Default is None"] = None,
    ) -> str:
        """Read the specified document."""
        with (working_directory / file_name).open("r") as file:
            lines = file.readlines()
        if start is not None:
            start = 0
        return "\n".join(lines[start:end])

    @tool
    def write_document(
        content: Annotated[str, "Text content to be written into the document."],
        file_name: Annotated[str, "File path to save the document."],
    ) -> Annotated[str, "Path of the saved document file."]:
        """Create and save a text document."""
        with (working_directory / file_name).open("w") as file:
            file.write(content)
        return f"Document saved to {file_name}"

    @tool
    def edit_document(
        file_name: Annotated[str, "Path of the document to be edited."],
        inserts: Annotated[
            Dict[int, str],
            "Dictionary where key is the line number (1-indexed) and value"
            " is the text to be inserted at that line.",
        ] = {},
    ) -> Annotated[str, "Path of the edited document file."]:
        """Edit a document by inserting text at specific line numbers."""

        with (working_directory / file_name).open("r") as file:
            lines = file.readlines()

        sorted_inserts = sorted(inserts.items())

        for line_number, text in sorted_inserts:
            if 1 <= line_number <= len(lines) + 1:
                lines.insert(line_number - 1, text + "\n")
            else:
                return f"Error: Line number {line_number} is out of range."

        with (working_directory / file_name).open("w") as file:
            file.writelines(lines)

        return f"Document edited and saved to {file_name}"

    def prelude(state):
        written_files = []
        if not working_directory.exists():
            working_directory.mkdir()
        try:
            written_files = [
                f.relative_to(working_directory) for f in working_directory.rglob("*")
            ]
        except Exception as e:
            print('ERROR', e)
            pass
        if not written_files:
            return {**state, "current_files": "No files written."}
        return {
            **state,
            "current_files": "\nBelow are files your team has written to the directory:\n"
            + "\n".join([f" - {f}" for f in written_files]),
        }

    checker_agent = create_agent(
        llm,
        [read_document, retrieve_information],
        ("You are an expert fact checking technical documents.\n{current_files}"),
    )
    context_aware_doc_writer_agent = prelude | checker_agent
    fact_checking_node = functools.partial(
        agent_node, agent=context_aware_doc_writer_agent, name="FactChecker"
    )

    editor_agent = create_agent(
        llm,
        [read_document, edit_document],
        ("You are an expert senior researcher tasked with fact_checking a LinkedIn post and"
        " updating information we be accurate and truthful.\n{current_files}"),
    )
    context_aware_note_taking_agent = prelude | editor_agent
    editor_node = functools.partial(
        agent_node, agent=context_aware_note_taking_agent, name="Editor"
    )

    supervisor = create_team_supervisor(
        llm,
        ("You are a supervisor tasked with managing a conversation between the"
        " following workers: {team_members}. You should always verify the technical"
        " contents after any edits are made."
        " Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When each team is finished,"
        " you must respond with FINISH."),
        ["FactChecker", "Editor"],
    )

    #####

    _fact_checking_graph = StateGraph(FactCheckingState)
    _fact_checking_graph.add_node("FactChecker", fact_checking_node)
    _fact_checking_graph.add_node("Editor", editor_node)
    _fact_checking_graph.add_node("supervisor", supervisor)

    _fact_checking_graph.add_edge("FactChecker", "supervisor")
    _fact_checking_graph.add_edge("Editor", "supervisor")

    _fact_checking_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "FactChecker": "FactChecker",
            "Editor": "Editor",
            "FINISH": END,
        },
    )

    _fact_checking_graph.set_entry_point("supervisor")
    fact_checking_graph = _fact_checking_graph.compile()

    #####
    # print_mermaid_image(fact_checking_graph, './images/fact_checking_graph.png')
    #####

    fact_checking_graph_chain = (
        functools.partial(enter_fact_checking_graph, team_members=fact_checking_graph.nodes)
        | fact_checking_graph
    )

    return fact_checking_graph_chain

    #####
    # for s in writing_graph_chain.stream(
    #     "Write an outline for for a short LinkedIn post on Linear Regression and write it to disk.",
    #     {"recursion_limit": 100},
    # ):
    #     if "__end__" not in s:
    #         print(s)
    #         print("---")
    #####
