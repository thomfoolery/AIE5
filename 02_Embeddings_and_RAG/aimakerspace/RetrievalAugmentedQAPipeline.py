from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    def run_pipeline(self, user_prompt: UserRolePrompt, system_prompt: SystemRolePrompt, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = system_prompt.create_message()
        formatted_user_prompt = user_prompt.create_message(user_query=user_query, context=context_prompt)

        return {"response" : self.llm.run([formatted_system_prompt, formatted_user_prompt]), "context" : context_list}