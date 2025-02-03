
import os

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

AGENT_SYSTEM_PROMPT = ("\nWork autonomously according to your specialty, using the tools available to you."
" Do not ask for clarification."
" Your other team members (and other teams) will collaborate with you with their own specialties."
" You are chosen for a reason!"
# " You are one of the following team members: {team_members}."
"")

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += AGENT_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

def create_working_directory(foldername: str = None) -> str:
    # random_id = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID
    # subdirectory_path = os.path.join('./content/data', random_id)
    basePath = Path('./workspace')
    os.makedirs(basePath, exist_ok=True)
    
    directory_path = os.path.join(basePath, foldername) if foldername else basePath

    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def print_mermaid_image(graph, path):
    try:
        image = graph.get_graph(xray=True).draw_mermaid_png()
        with open(path, 'wb') as f:
            f.write(image)
    except Exception as e:
        print('Error', e)
        pass