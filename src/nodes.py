import asyncio
from typing import Any, Coroutine, cast

from langchain.messages import RemoveMessage
from langchain_core.documents.base import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from langsmith import traceable

from src import create_logger
from src.config import app_config
from src.prompts import PromptsBuilder
from src.schemas.nodes_schema import (
    Plan,
    ReWrittenQuery,
    Step,
    StructuredMemoryResponse,
    ValidateQuery,
)
from src.schemas.types import (
    MemoryKeys,
    NextAction,
    SectionNamesType,
    SummarizationConditionType,
    ToolsType,
)
from src.state import State
from src.utilities.llm_utils import remote_llm
from src.utilities.tools.tools import (
    arerank_documents,
    atavily_web_search_tool,
)
from src.utilities.utils import (
    append_memory,
    aretrieve_internal_documents,
    convert_context_to_str,
    convert_langchain_messages_to_dicts,
    determine_retrieval_type,
    format_documents,
    format_plan,
    get_decision,
    get_structured_output,
    query_rewriter,
)

logger = create_logger("nodes")

# Constants
FETCH_FULL_PAGE: bool = app_config.custom_config.fetch_full_page
K: int = app_config.custom_config.k
RERANK_K: int = app_config.custom_config.rerank_k
MAX_MESSAGES: int = app_config.custom_config.max_messages
MAX_CHARS: int | None = app_config.custom_config.max_chars
MAX_ATTEMPTS: int = app_config.custom_config.max_attempts


prompt_builder = PromptsBuilder()
section_titles: list[str] = [s.value for s in SectionNamesType]


# =========================================================
# ========================= NODES =========================
# =========================================================
@traceable
async def validate_query_node(state: State) -> dict[str, Any]:
    """Validate the user's query to ensure it is relevant to the specified topics.

    Parameters
    ----------
    state : State
        Current state containing the original_question.

    Returns
    -------
    dict[str, Any]
        Updated state with validation results.
    """
    topics: str = " | ".join(app_config.custom_config.topics)
    user_question: str = state["original_question"]
    user_query: str = f"<USER_QUESTION>{user_question}</USER_QUESTION>"
    sys_msg = prompt_builder.query_validation_prompt(topics=topics)
    messages = convert_langchain_messages_to_dicts(
        messages=[SystemMessage(content=sys_msg), HumanMessage(content=user_query)]
    )
    logger.info("🚨 Validating user question against context topics...")
    response = await get_structured_output(
        messages=messages, model=None, schema=ValidateQuery
    )
    response = cast(ValidateQuery, response)
    # logger.info(
    #     f"🚨 Related to topic?: {response.is_related_to_context} | "
    #     f"Next Action: {response.next_action} | Rationale: {response.rationale}"
    # )

    step_state: dict[str, Any] = {
        "step_index": -1,
        "question": user_question,
        "rewritten_queries": [],
        "reranked_documents": [],
        "summary": response.rationale,
    }

    return {
        "current_step_index": -1,
        "is_related_to_context": response.is_related_to_context,
        "step_state": [step_state],
        "plan": None,
        "synthesized_context": state.get("synthesized_context", ""),
    }


@traceable
async def generate_plan_node(
    state: State, config: RunnableConfig, store: BaseStore
) -> dict[str, Any]:
    """Generate a multi-step plan based on the user's question.

    Parameters
    ----------
    state : State
        Current state containing the original_question.
    config : RunnableConfig
        Configuration for the runnable, including user_id.
    store : BaseStore
        Store for retrieving user preferences and memory.

    Returns
    -------
    dict[str, Any]
        Updated state with the generated plan.
    """
    # If plan already exists AND we're not re-planning, return empty update (no overwrite)
    # Re-planning is detected when all steps are completed (current_step_index >= num_steps)
    current_plan = state.get("plan")
    current_step_idx = state.get("current_step_index", 0)

    is_replanning: bool = bool(
        current_plan and (current_step_idx >= len(current_plan.steps))
    )

    if current_plan and not is_replanning:
        logger.info("Plan already exists. Skipping plan generation.")
        return {}

    # Build context for re-planning
    replan_context: str = ""
    if is_replanning:
        logger.info("🔄 Re-planning: Previous plan completed but research incomplete.")
        history: str = convert_context_to_str(state["step_state"])
        replan_context = f"""
        <PREVIOUS_RESEARCH_SUMMARY>
            The initial plan was completed but research is incomplete. Here's what was already
            investigated:
            <HISTORY>{history}</HISTORY>
            Generate a NEW refined plan with different approaches or angles to address remaining
            gaps.
        </PREVIOUS_RESEARCH_SUMMARY>
        """

    summary: str = state.get("conversation_summary", "")
    user_question: str = state.get("original_question", "")
    user_query: str = f"<USER_QUESTION>{user_question}</USER_QUESTION>{replan_context}"

    # ============== Process Long-term Memory ================
    user_id: str = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace_key: str = MemoryKeys.NAMESPACE_KEY.value
    namespace: tuple[str, str] = (namespace_key, user_id)
    key: str = MemoryKeys.USER_PREFERENCES_KEY.value
    user_preferences = await store.aget(namespace, key)

    # Extract memory if it exists
    if user_preferences:
        user_preferences_content: str = user_preferences.value.get(namespace_key)
    else:
        user_preferences_content = "No memory found."

    sys_msg = prompt_builder.planner_prompt(
        section_titles=" | ".join(section_titles),
        summary=summary,
        user_preferences_content=user_preferences_content,
    )

    messages = convert_langchain_messages_to_dicts(
        messages=[SystemMessage(content=sys_msg), HumanMessage(content=user_query)]
    )
    response = await get_structured_output(messages=messages, model=None, schema=Plan)
    plan_formatted: str = f"Multi-steps Plan:\n{format_plan(plan=response)}"
    log_msg: str = f"Number of steps: {len(response.steps)}..."
    if is_replanning:
        log_msg = f"🔄 RE-PLAN Generated. {log_msg}"
    logger.info(log_msg)

    return {
        "is_related_to_context": True,
        "plan": response,
        "step_state": [None],  # Reset signal - clear when starting new plan
        "current_step_index": 0,
        "messages": [user_question, plan_formatted],
        "retrieved_documents": [None],  # Reset signal - clear previous retrieval
        "reranked_documents": [None],  # Reset signal - clear previous reranking
        "synthesized_context": "",  # Clear previous synthesis
        "conversation_summary": state.get("conversation_summary", ""),
    }


@traceable
async def retrieve_internal_docs_node(state: State) -> dict[str, Any]:
    """Retrieve internal documents node."""

    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]
    logger.info(
        f"🛢 Using Vector DB\nRetrieving documents for Step "
        f"{current_step_idx}: {current_step.question}"
    )

    # Re-write the query and determine retrieval method concurrently
    re_written_query_obj, retriever_method = await asyncio.gather(
        query_rewriter(
            question=current_step.question,
            search_keywords=current_step.search_keywords,
        ),
        determine_retrieval_type(question=current_step.question),
    )

    rewritten_queries: list[str] = re_written_query_obj.rewritten_query
    # logger.info(f"Re-written queries: {rewritten_queries}")
    # logger.info(
    #     f"Selected retrieval method: {retriever_method.method};"
    #     f"\nRationale: {retriever_method.rationale}"
    # )

    # Retrieve documents based on the selected method
    retrieved_docs: list[Document] = await aretrieve_internal_documents(
        method=retriever_method.method,
        rewritten_queries=rewritten_queries,
        target_section=current_step.target_section,
        k=K,
    )
    reranked_docs: list[Document] = await arerank_documents(
        query=current_step.question, documents=retrieved_docs, k=RERANK_K
    )
    step_state = {
        "step_index": current_step_idx,
        "question": current_step.question,
        "rewritten_queries": rewritten_queries,
        "reranked_documents": reranked_docs,
        "summary": "",
    }
    # Update the state with retrieved docs and step_state
    # Append to existing step_state to accumulate history
    return {
        "step_state": state.get("step_state", []) + [step_state],
        "retrieved_documents": retrieved_docs,
        "reranked_documents": reranked_docs,
    }


@traceable
async def internet_search_node(state: State) -> dict[str, Any]:
    """Retrieve documents from the web using re-written queries.

    Parameters
    ----------
    state : State
        Current state of the agent.

    Returns
    -------
    dict[str, Any]
        The retrieved documents
    """
    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]
    logger.info(
        f"Retrieving documents for Step {current_step_idx}: {current_step.question}"
    )

    # Re-write the query using the query re-writer
    re_written_query_obj: ReWrittenQuery = await query_rewriter(
        question=current_step.question,
        search_keywords=current_step.search_keywords,
    )
    rewritten_queries: list[str] = re_written_query_obj.rewritten_queries
    # logger.info(f"🌐 WEB SEARCH\nRe-written queries: {rewritten_queries}")

    tasks: list[Coroutine[Any, Any, list[Document]]] = [
        atavily_web_search_tool(
            query=query,
            fetch_full_page=FETCH_FULL_PAGE,
            k=K,
            max_chars=MAX_CHARS,
        )
        for query in rewritten_queries
    ]
    all_docs: list[list[Document]] = await asyncio.gather(*tasks)
    # Flatten the docs
    retrieved_docs: list[Document] = [doc for sublist in all_docs for doc in sublist]  # type: ignore
    reranked_docs: list[Document] = await arerank_documents(
        query=current_step.question, documents=retrieved_docs, k=RERANK_K
    )

    step_state = {
        "step_index": current_step_idx,
        "question": current_step.question,
        "rewritten_queries": rewritten_queries,
        "reranked_documents": reranked_docs,
        "summary": "",
    }

    # Update the state with retrieved docs and step_state
    # Append to existing step_state to accumulate history
    return {
        "step_state": state.get("step_state", []) + [step_state],
        "retrieved_documents": retrieved_docs,
        "reranked_documents": reranked_docs,
    }


@traceable
async def compress_documents_node(state: State) -> dict[str, Any]:
    """Compress reranked documents for each step."""

    reranked_documents: list[Document] = state["reranked_documents"]
    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]

    # Format document
    user_query: str = f"<DOCUMENT>{format_documents(reranked_documents)}</DOCUMENT>"
    sys_msg: str = prompt_builder.compression_prompt(question=current_step.question)
    logger.info(
        f"Synthesizing documents for Step {current_step_idx}: {current_step.question}"
    )
    response = await remote_llm.ainvoke(
        [SystemMessage(sys_msg), HumanMessage(user_query)]
    )

    return {"step_index": current_step_idx, "synthesized_context": response.content}


@traceable
async def reflection_node(state: State) -> dict[str, Any]:
    """This node is responsible for moving to the next step in the multi-step plan."""

    synthesized_context: str = state["synthesized_context"]
    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]

    logger.info(f"Summarizing for Step {current_step_idx}: {current_step.question}")
    new_step_state = {
        "step_index": current_step_idx,
        "summary": synthesized_context,
    }
    logger.info(
        f"⚠️ Number of steps completed: {current_step_idx + 1} | Num iterations: "
        f"{state.get('num_iterations', 0) + 1}"
    )

    return {
        "current_step_index": current_step_idx + 1,
        # Append the new step state to the existing list
        "step_state": state.get("step_state", []) + [new_step_state],
        # Keep all messages for context
        "messages": state.get("messages", []) + [synthesized_context],
        # Move to the next step
        "num_iterations": state.get("num_iterations", 0) + 1,
    }


@traceable
async def final_answer_node(state: State) -> dict[str, Any]:
    """Generate the final answer with citations based on all collected evidence."""

    logger.info("--- ✅: Generating Final Answer with Citations ---")
    # Gather all the evidence we've collected from ALL steps.
    final_context: str = ""
    for step in state["step_state"]:
        # Skip the validation step (step_index == -1)
        if step.get("step_index") == -1:
            continue
        final_context += (
            f"\n--- Findings from Research Step {step['step_index'] + 1} ---\n"
        )
        # Use reranked_documents which exists in step_state
        for doc in step.get("reranked_documents", []):  # type: ignore
            source: str = (  # type: ignore
                f"Filename: {doc.metadata.get('source')} | Section: {doc.metadata.get('section')}"
                or doc.metadata.get("url")
            )
            final_context += f"Source: {source}\nContent: {doc.page_content}\n\n"

    prompt: str = prompt_builder.final_answer_prompt(
        question=state["original_question"]
    )
    context: str = f"<CONTEXT>{final_context}</CONTEXT>"

    final_answer = await remote_llm.ainvoke(
        [SystemMessage(prompt), HumanMessage(context)]
    )
    # Update the state with the final answer and reset ALL accumulating state
    # Use None sentinel to trigger reset in custom reducers
    return {
        "final_answer": final_answer.content,
        "plan": None,  # Clear plan
        "step_state": [None],  # Reset signal for merge_step_states reducer
        "retrieved_documents": [None],  # Reset signal for merge_documents reducer
        "reranked_documents": [None],  # Reset signal for merge_documents reducer
        "synthesized_context": "",  # Clear synthesis
        "current_step_index": -1,  # Reset step counter
        "num_iterations": 0,  # Reset iteration counter
    }


def unrelated_query_node(state: State) -> dict[str, Any]:  # noqa: ARG001
    """Handle unrelated queries by providing a default response.

    Parameters
    ----------
    state : State
        The current state of the agent.

    Returns
    -------
    dict[str, Any]
        Updated state with a default final answer.
    """
    logger.info("🚨 Query unrelated to context. Generating default response...")
    default_response: str = (
        "I'm sorry, but your question does not relate to the available information "
        "about NVIDIA's financial performance, form 10-K, news related to NVIDIA, "
        "or industry trends. Please ask a question relevant to these topics."
    )
    return {
        "plan": None,
        "final_answer": default_response,
        "step_state": [None],  # Reset signal for merge_step_states reducer
        "retrieved_documents": [None],  # Reset signal for merge_documents reducer
        "reranked_documents": [None],  # Reset signal for merge_documents reducer
        "synthesized_context": "",  # Clear synthesis
        "current_step_index": -1,  # Reset step counter
        "num_iterations": 0,  # Reset iteration counter
    }


async def overall_convo_summarization_node(state: State) -> dict[str, Any]:
    """Summarization node to condense the conversation history."""
    summary: str = state.get("conversation_summary", "")

    if summary:
        summary_msg: list[AnyMessage] = [
            HumanMessage(
                content=prompt_builder.overall_convo_summary_prompt(summary=summary)
            )
        ]
    else:
        summary_msg = [
            HumanMessage(content=prompt_builder.no_existing_summary_prompt())
        ]

    try:
        response: AIMessage = await remote_llm.ainvoke(state["messages"] + summary_msg)
        logger.info("✅ Conversation history summarized.")

    except Exception as e:
        logger.error(f"⚠️ Error in summarization LLM call: {e}")
        # Fallback to returning the existing summary if summarization fails
        return {"summary": summary}

    # Delete ALL but the last 2 messages
    messages_to_remove: list[AnyMessage] = [
        RemoveMessage(id=m.id) for m in state["messages"][:-2]
    ]  # type: ignore

    return {
        # The `add_messages` reducer will handle removing the old messages
        "messages": messages_to_remove,
        # Add the new summary to the state
        "conversation_summary": response.content,
    }


async def update_lt_memory_node(
    state: State, config: RunnableConfig, store: BaseStore
) -> None:
    """Update user (long-term) memory based on the conversation."""
    try:
        user_id: str = config["configurable"]["user_id"]
        namespace_key: str = MemoryKeys.NAMESPACE_KEY.value
        namespace: tuple[str, str] = (namespace_key, user_id)
        key: str = MemoryKeys.USER_PREFERENCES_KEY.value

        # Get existing memory
        user_preferences_content = await store.aget(namespace, key)
        existing_memory: dict[str, Any] = (
            user_preferences_content.value.get(namespace_key, {})
            if user_preferences_content
            else {}
        )

        # Format for prompt (convert dict to readable string)
        if existing_memory:
            formatted: str = "\n".join(
                f"- {k}: {v}" for k, v in existing_memory.items() if v
            )
        else:
            formatted = "No memory found."

        sys_msg: str = prompt_builder.update_user_memory_prompt(
            user_preferences_content=formatted
        )
        summary: str = state.get("conversation_summary", "")

        # Build context
        context = [SystemMessage(content=sys_msg)]
        if summary:
            context.append(SystemMessage(content=f"Summary: {summary}"))
        # Add recent messages
        context.extend(state["messages"])

        try:
            messages: list[dict[str, str]] = convert_langchain_messages_to_dicts(
                context
            )
            new_memory: StructuredMemoryResponse = await get_structured_output(  # type: ignore
                messages=messages,
                model=None,
                schema=StructuredMemoryResponse,
            )
        except Exception as e:
            logger.error(f"⚠️ Error in memory update: {e}")
            return

        if new_memory:
            # Simple append: existing + new
            updated_memory: dict[str, Any] = append_memory(  # type: ignore
                existing_memory,
                new_memory.model_dump(),
            )
            await store.aput(namespace, key, value={"memory": updated_memory})
            logger.info("💥 Memory updated")

    except Exception as e:
        logger.error(f"⚠️ Error in update_memory_node: {e}")

    return


# =========================================================
# =================== CONDITIONAL NODES ===================
# =========================================================
def route_by_tool_condition(state: State) -> ToolsType:
    """Determine the tool type for the current step.

    Parameters
    ----------
    state : State
        The current state of the agent.

    Returns
    -------
    ToolsType
        The tool type for the current step.
    """
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]
    return current_step.tool


def should_summarize_overal_convo(state: State) -> SummarizationConditionType:
    """Edge to determine if summarization is needed."""
    if len(state["messages"]) > MAX_MESSAGES:
        return SummarizationConditionType.SUMMARIZE

    return SummarizationConditionType.END


async def should_continue_condition(
    state: State, max_reasoning_interations: int = 8
) -> NextAction:
    """Determine if the current step should be retried, re-planned, or finished.

    Parameters
    ----------
    state : State
        The current state of the agent.
    max_reasoning_interations : int, optional
        The maximum number of reasoning iterations allowed, by default 8.

    Returns
    -------
    NextAction
        The next action to take (CONTINUE, FINISH, or RE_PLAN).
    """
    logger.info("--- Evaluating Multi Step Reasoning Policy ---")
    is_related_to_context: bool = state.get("is_related_to_context", True)
    current_step_idx: int = state["current_step_index"]
    num_iterations: int = state.get("num_iterations", 0)

    # ===== Checks =====
    # If query does NOT relate to the topics, finish immediately
    if not is_related_to_context:
        logger.info(" -> Query not related to context. Finishing...")
        return NextAction.FINISH

    # Are all the steps completed?
    if state["plan"] and (current_step_idx >= len(state["plan"].steps)):
        logger.info(
            " -> Plan complete. All steps are done. Checking if research is complete..."
        )

        # Get decision from LLM to determine if we should re-plan or finish
        history: str = convert_context_to_str(state["step_state"])
        decision = await get_decision(
            question=state["original_question"],
            plan=state["plan"],
            history=history,
        )
        logger.info(
            f" -> Decision: {decision.next_action} | Rationale: {decision.rationale}"
        )

        if decision.next_action == NextAction.RE_PLAN:
            logger.info(
                "📋 -> Re-planning: Research incomplete, generating new plan..."
            )
            return NextAction.RE_PLAN
        return NextAction.FINISH

    # Is the max num of iterations exhausted?
    if num_iterations >= max_reasoning_interations:
        logger.info(
            f" -> Max iterations reached. {num_iterations} iterations. Finishing..."
        )
        return NextAction.FINISH

    # No retrieved documents yet
    if state.get("reranked_documents") is not None and not state["reranked_documents"]:
        logger.info(
            "⚠️ -> No retrieved documents yet. Continuing with next step in plan..."
        )
        return NextAction.CONTINUE

    # If the conditions above are NOT met
    history: str = convert_context_to_str(state["step_state"])

    # Get decision from LLM
    decision = await get_decision(
        question=state["original_question"],
        plan=state["plan"],
        history=history,
    )
    logger.info(
        f" -> Decision: {decision.next_action} | Rationale: {decision.rationale}"
    )

    if decision.next_action == NextAction.FINISH:
        return NextAction.FINISH
    return NextAction.CONTINUE
