"""Example code for planner-worker agent collaboration with multiple tools."""

import asyncio
import contextlib
import signal
import sys

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client
from src.utils.tools.gemini_grounding import (
    GeminiGroundingWithGoogleSearch,
    ModelSettings,
)


load_dotenv(verbose=True)

set_up_logging()

AGENT_LLM_NAMES = {
    "worker": "gemini-2.5-flash",  # less expensive,
    "planner": "gemini-2.5-pro",  # more expensive, better at reasoning and planning
}

configs = Configs.from_env_var()
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="recipes",
)

gemini_grounding_tool = GeminiGroundingWithGoogleSearch(
    model_settings=ModelSettings(model=AGENT_LLM_NAMES["worker"])
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


# Worker Agent: handles long context efficiently
kb_agent = agents.Agent(
    name="Naveen_KnowledgeBaseAgent",
    instructions="""
        You are an agent specialized in searching a knowledge base.
        You will receive a single search query as input.
        Use the 'search_knowledgebase' tool to perform a search, then return a
        JSON object with:
        - 'Summary': Name of the Recipe found with the description provided
        - 'Sugar' : Amount of Sugar in Each Recipe
        - 'Ingredients' : All Ingredients in the Recipe
        If the tool returns no matches, set "no_results": true and keep "sources" empty.
        Do NOT make up information. Do NOT return raw search results or long quotes.
    """,
    tools=[
        agents.function_tool(async_knowledgebase.search_knowledgebase),
    ],
    # a faster, smaller model for quick searches
    model=agents.OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["worker"], openai_client=async_openai_client
    ),
)

# Main Agent: more expensive and slower, but better at complex planning
main_agent = agents.Agent(
    name="MainAgent",
    instructions="""
        You are a deep research agent and your goal is to conduct in-depth, multi-turn
        research by breaking down complex queries, using the provided tools, and
        synthesizing the information into a comprehensive report.

        You have access to the following tools:
        1. 'search_knowledgebase' - use this tool to search for information in a
            knowledge base. The knowledge base reflects recipe dataset from a select set of recipes. The recipe 
            containts information like recipe_name,	prep_time,	cook_time,	total_time,	servings,	yield,	
            ingredients	directions,	rating,	url,	cuisine_path,	nutrition,
            timing

        2. 'get_web_search_grounded_response' - use this tool for current events,
            news, and checking which celebrities prefer eating some of the recipies which were searched. 
            Don't look for Exact Match of the recipe , a close overall description is fine 

        Both tools will not return raw search results or the sources themselves.
        Instead, they will return a concise summary of the key findings, along
        with the sources used to generate the summary.

        For best performance, divide complex queries into simpler sub-queries
        Before calling either tool, always explain your reasoning for doing so.

        Note that the 'get_web_search_grounded_response' tool will expand the query
        into multiple search queries and execute them. It will also return the
        queries it executed. Do not repeat them.

        **Guidelines for synthesis**
        - After collecting results, write the final answer from your own synthesis.
        - The response should be formatted as  as:
            [1] Recipe Name and Summary
            [2] Celbrities loving those recipies. A Clickable URL from where celebrities response was collected.
            [2] Sugar contect in recipe
            [3] Ingredients
        - Do not invent URLs or sources.
        - If both tools fail, say so and suggest 2–3 refined queries.

        Be sure to mention the sources in your response, including the URL if available,
        and do not make up information.
    """,
    # Allow the planner agent to invoke the worker agent.
    # The long context provided to the worker agent is hidden from the main agent.
    tools=[
        kb_agent.as_tool(
            tool_name="search_recipe",
            tool_description=(
                "Search the knowledge base for a query and return a concise summary "
                "of the key findings, along with the sources used to generate "
                "the summary. Response with two recipes always"
            ),
        ),
        agents.function_tool(gemini_grounding_tool.get_web_search_grounded_response),
    ],
    # a larger, more capable model for planning and reasoning over summaries
    model=agents.OpenAIChatCompletionsModel(
        model=AGENT_LLM_NAMES["planner"], openai_client=async_openai_client
    ),
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    # Use the main agent as the entry point- not the worker agent.
    with langfuse_client.start_as_current_span(name="Naveen_Search_Agent_Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)


demo = gr.ChatInterface(
    _main,
    title="2.3 Multi-Agent with Multiple Search Tools",
    type="messages",
    examples=[
            "Show me a recipe made up of apples ",
            " Show me a recipe which takes less than 15 minutes",
    ],
)

if __name__ == "__main__":
    async_openai_client = AsyncOpenAI()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
