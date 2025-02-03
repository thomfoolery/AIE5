from typing import Dict, Any, Callable, List, Optional, TypedDict, Union, Annotated, Tuple
from typing_extensions import TypedDict
import functools
import operator

from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from agents.utils import agent_node, create_agent, create_team_supervisor

class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str

def enter_writing_graph(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results

def create_writing_team(llm: ChatOpenAI, working_directory) -> str:

    @tool
    def create_outline(
        points: Annotated[List[str], "List of main points or sections."],
        file_name: Annotated[str, "File path to save the outline."],
    ) -> Annotated[str, "Path of the saved outline file."]:
        """Create and save an outline."""
        with (working_directory / file_name).open("w") as file:
            for i, point in enumerate(points):
                file.write(f"{i + 1}. {point}\n")
        return f"Outline saved to {file_name}"

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
            "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
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

    doc_writer_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        ("You are an expert writing technical LinkedIn posts.\n"
        "Below are files currently in your directory:\n{current_files}"),
    )
    context_aware_doc_writer_agent = prelude | doc_writer_agent
    doc_writing_node = functools.partial(
        agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
    )

    note_taking_agent = create_agent(
        llm,
        [create_outline, read_document],
        ("You are an expert senior researcher tasked with writing a LinkedIn post outline and"
        " taking notes to craft a LinkedIn post.\n{current_files}"),
    )
    context_aware_note_taking_agent = prelude | note_taking_agent
    note_taking_node = functools.partial(
        agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
    )

    copy_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        ("You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues\n"
        "Below are files currently in your directory:\n{current_files}"),
    )
    context_aware_copy_editor_agent = prelude | copy_editor_agent
    copy_editing_node = functools.partial(
        agent_node, agent=context_aware_copy_editor_agent, name="CopyEditor"
    )

    dopeness_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        ("You are an expert in dopeness, litness, coolness, etc - you edit the document to make sure it's dope. Make sure to use a number of emojis."
        "Below are files currently in your directory:\n{current_files}"),
    )
    context_aware_dopeness_editor_agent = prelude | dopeness_editor_agent
    dopeness_node = functools.partial(
        agent_node, agent=context_aware_dopeness_editor_agent, name="DopenessEditor"
    )

    doc_writing_supervisor = create_team_supervisor(
        llm,
        ("You are a supervisor tasked with managing a conversation between the"
        " following workers: {team_members}. You should always verify the technical"
        " contents after any edits are made. "
        "Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When each team is finished,"
        " you must respond with FINISH."),
        ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"],
    )

    #####

    _writing_graph = StateGraph(DocWritingState)
    _writing_graph.add_node("DocWriter", doc_writing_node)
    _writing_graph.add_node("NoteTaker", note_taking_node)
    _writing_graph.add_node("CopyEditor", copy_editing_node)
    _writing_graph.add_node("DopenessEditor", dopeness_node)
    _writing_graph.add_node("supervisor", doc_writing_supervisor)

    _writing_graph.add_edge("DocWriter", "supervisor")
    _writing_graph.add_edge("NoteTaker", "supervisor")
    _writing_graph.add_edge("CopyEditor", "supervisor")
    _writing_graph.add_edge("DopenessEditor", "supervisor")

    _writing_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "DocWriter": "DocWriter",
            "NoteTaker": "NoteTaker",
            "CopyEditor" : "CopyEditor",
            "DopenessEditor" : "DopenessEditor",
            "FINISH": END,
        },
    )

    _writing_graph.set_entry_point("supervisor")
    writing_graph = _writing_graph.compile()

    #####
    # Print Mermaid Image
    # try:
    #     image = writing_graph.get_graph(xray=True).draw_mermaid_png()
    #     with open('./images/writing_graph.png', 'wb') as f:
    #         f.write(image)
    # except Exception as e:
    #     print('Error', e)
    #     pass
    #####

    writing_graph_chain = (
        functools.partial(enter_writing_graph, members=writing_graph.nodes)
        | writing_graph
    )

    return writing_graph_chain

    #####
    # for s in writing_graph_chain.stream(
    #     "Write an outline for for a short LinkedIn post on Linear Regression and write it to disk.",
    #     {"recursion_limit": 100},
    # ):
    #     if "__end__" not in s:
    #         print(s)
    #         print("---")
    #####
