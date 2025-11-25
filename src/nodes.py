import asyncio
import json
from typing import Any, Callable, Coroutine, cast

from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from src import create_logger
from src.config import app_config
from src.prompts import PromptsBuilder
from src.schemas.nodes_schema import (
    Decision,
    Plan,
    RetrieverMethod,
    ReWrittenQuery,
    Step,
    ValidateQuery,
)
from src.schemas.types import NextAction, RetrieverMethodType, ToolsType
from src.state import State, StepState
from src.utilities.llm_utils import remote_llm
from src.utilities.tools.tools import (
    ahybrid_search_tool,
    akeyword_search_tool,
    arerank_documents,
    atavily_web_search_tool,
    avector_search_tool,
)
from src.utilities.utils import (
    convert_langchain_messages_to_dicts,
    get_structured_output,
)

logger = create_logger("nodes")

# Constants
FETCH_FULL_PAGE: bool = app_config.custom_config.fetch_full_page
K: int = app_config.custom_config.k
MAX_CHARS: int | None = app_config.custom_config.max_chars
MAX_ATTEMPTS: int = app_config.custom_config.max_attempts

type RetrieverFn = Callable[[str, str | None, int], Coroutine[Any, Any, list[Document]]]
retrieval_method_dicts: dict[str, RetrieverFn] = {
    RetrieverMethodType.VECTOR_SEARCH: avector_search_tool,
    RetrieverMethodType.KEYWORD_SEARCH: akeyword_search_tool,
    RetrieverMethodType.HYBRID_SEARCH: ahybrid_search_tool,
}
prompt_builder = PromptsBuilder()
section_titles: list[str] = [
    "ITEM 1. BUSINESS",
    "ITEM 1A. RISK FACTORS",
    "ITEM 1B. UNRESOLVED STAFF COMMENTS",
    "ITEM 2. PROPERTIES",
    "ITEM 3. LEGAL PROCEEDINGS",
    "ITEM 4. MINE SAFETY DISCLOSURES",
    "ITEM 5. MARKET FOR REGISTRANT’S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES",
    "ITEM 6. [RESERVED]",
    "ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
    "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
    "ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE",
    "ITEM 9A. CONTROLS AND PROCEDURES",
    "ITEM 9C. DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPECTIONS",
    "ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE",
    "ITEM 11. EXECUTIVE COMPENSATION",
    "ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND RELATED STOCKHOLDER MATTERS",
    "ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS, AND DIRECTOR INDEPENDENCE",
    "ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES",
    "ITEM 15. EXHIBIT AND FINANCIAL STATEMENT SCHEDULES",
    "ITEM 16. FORM 10-K SUMMARY",
]


# =========================================================
# ============== HELPER FUNCTIONS FOR NODES ===============
# =========================================================
def deduplicate(documents: list[Document]) -> list[Document]:
    """Deduplicate documents based on 'chunk_id' in metadata."""
    docs_dict: dict[str, Document] = {}

    if not documents[0].metadata:
        raise ValueError(
            "Cannot deduplicate documents without 'chunk_id' in metadata. Please ensure documents have "
            "'chunk_id' in their metadata."
        )
    for doc in documents:
        if (_id := doc.metadata["chunk_id"]) not in docs_dict:
            docs_dict[_id] = doc
    return list(docs_dict.values())


def format_documents(documents: list[Document]) -> str:
    """Format documents for synthesis input."""
    delimiter: str = "===" * 20
    try:
        docs: list[str] = [
            f"[Source]: {doc.metadata['source_doc']}\n[Content]: {doc.page_content}\n{delimiter}"
            for doc in documents
        ]
    except KeyError:
        docs = [
            f"[Source]: {doc.metadata['url']}\n[Content]: {doc.page_content}\n{delimiter}"
            for doc in documents
        ]
    formated_docs: str = "\n\n".join(docs)

    return formated_docs


async def aretrieve_internal_documents(
    method: RetrieverMethodType | str,
    rewritten_queries: list[str],
    target_section: str | None,
    k: int,
) -> list[Document]:
    """Retrieve internal documents using the specified retrieval method.

    Parameters
    ----------
    method : RetrieverMethodType | str
        Retrieval method to use (`vector_search`, `keyword_search`, or `hybrid_search`).
    rewritten_queries : list[str]
        Query variations produced by the query rewriter for this step.
    target_section : str | None
        Target section filter for internal document search. Only applied when
        method is `vector_search` or `hybrid_search`; ignored for pure keyword search.
    k : int
        Number of top documents to retrieve per query before deduplication.

    Returns
    -------
    list[Document]
        List of unique retrieved documents across all query variations.

    Raises
    ------
    ValueError
        If the provided method is not supported.
    """
    method = (
        method
        if isinstance(method, RetrieverMethodType)
        else RetrieverMethodType(method)
    )
    retrieval_fn = retrieval_method_dicts.get(method)
    if retrieval_fn is None:
        raise ValueError(f"Unsupported retrieval method: {method}")

    tasks: list[Coroutine[Any, Any, list[Document]]] = [
        # Expected signature. Order: query, target_section, k is important!
        retrieval_fn(
            query,
            target_section,
            k,
        )
        for query in rewritten_queries
    ]
    all_docs: list[list[Document]] = await asyncio.gather(*tasks)
    # Flatten the docs
    retrieved_docs: list[Document] = [doc for sublist in all_docs for doc in sublist]  # type: ignore

    return deduplicate(documents=retrieved_docs)


async def query_rewriter(question: str, search_keywords: list[str]) -> ReWrittenQuery:
    """Re-write the user's question into multiple query variations."""
    prompt = prompt_builder.query_rewriter_prompt(
        question=question, search_keywords=", ".join(search_keywords)
    )
    messages = convert_langchain_messages_to_dicts(messages=[HumanMessage(prompt)])
    response = await get_structured_output(
        messages=messages, model=None, schema=ReWrittenQuery
    )
    return cast(ReWrittenQuery, response)


async def determine_retrieval_type(question: str) -> RetrieverMethod:
    """Determine the optimal retrieval method for the given question."""
    prompt = prompt_builder.retriever_type_prompt(question=question)
    messages = convert_langchain_messages_to_dicts(messages=[HumanMessage(prompt)])
    response = await get_structured_output(
        messages=messages, model=None, schema=RetrieverMethod
    )
    return cast(RetrieverMethod, response)


def convert_context_to_str(state_state: list[StepState]) -> str:
    """This function converts the list of StepState dictionaries into a single string.

    Parameters
    ----------
    state_state : list[StepState]
        The list of StepState dictionaries representing the research history.

    Returns
    -------
    str
        A single string representation of the research history.
    """
    return "\n\n".join(
        [
            f"Step {s['step_index']}: {s['question']}\nSummary: {s['summary']}"
            for s in state_state
        ]
    )


def format_plan(plan: Plan | None) -> str:
    """Format the plan into a string representation.

    Parameters
    ----------
    plan : Plan
        The multi-step plan to be formatted.

    Returns
    -------
    str
        A string representation of the plan.
    """
    if plan is None:
        return ""
    return json.dumps([step.model_dump() for step in plan.steps])


async def get_decision(question: str, plan: Plan | None, history: str) -> Decision:
    """This node is used to determine whether to continue with the plan or finish.

    Parameters
    ----------
    question : str
        The original user question.
    plan : Plan
        The multi-step plan object.
    history : str
        The history of completed steps.

    Returns
    -------
    Decision
        The decision object containing the next action and rationale.
    """
    sys_msg = prompt_builder.decision_prompt(
        question=question, plan=format_plan(plan=plan)
    )
    history_query: str = f"<COMPLETED_STEPS>{history}</COMPLETED_STEPS>"

    messages: list[dict[str, str]] = convert_langchain_messages_to_dicts(
        messages=[SystemMessage(sys_msg), HumanMessage(history_query)]
    )
    response = await get_structured_output(
        messages=messages, model=None, schema=Decision
    )
    return cast(Decision, response)


async def rerank_retrieved_documents(state: State) -> dict[str, Any]:
    """Rerank documents by relevance to query."""
    k: int = 3
    question: str = state["original_question"]
    retrieved_documents: list[Document] = state["retrieved_documents"]
    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]
    logger.info(
        f"Retrieving documents for reranking for Step {current_step_idx}: {current_step.question}"
    )

    reranked_docs: list[Document] = await arerank_documents(
        query=question, documents=retrieved_documents, k=k
    )

    return {"reranked_documents": reranked_docs}


async def compression_documents(state: State) -> dict[str, Any]:
    """Synthesize final context from reranked documents."""
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

    return {"synthesized_context": response.content}


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
    logger.info(
        f"🚨 Related to topic?: {response.is_related_to_context} | "
        f"Next Action: {response.next_action} | Rationale: {response.rationale}"
    )

    step_state: list[StepState] = [
        StepState(
            step_index=-1,
            question=user_question,
            rewritten_queries=[],
            retrieved_documents=[],
            summary=response.rationale,
        )
    ]

    return {
        "current_step_index": -1,
        "is_related_to_context": response.is_related_to_context,
        "step_state": step_state,
        "plan": None,
    }


@traceable
async def generate_plan_node(state: State) -> dict[str, Any]:
    """Generate a multi-step plan based on the user's question.

    Parameters
    ----------
    state : State
        Current state containing the original_question.

    Returns
    -------
    dict[str, Any]
        Updated state with the generated plan.
    """
    # If plan already exists, return empty update (no overwrite)
    if state.get("plan"):
        return {}

    user_question: str = state["original_question"]
    user_query: str = f"<USER_QUESTION>{user_question}</USER_QUESTION>"

    sys_msg = prompt_builder.planner_prompt(section_titles=" | ".join(section_titles))

    messages = convert_langchain_messages_to_dicts(
        messages=[SystemMessage(content=sys_msg), HumanMessage(content=user_query)]
    )
    response = await get_structured_output(
        messages=messages, model="x-ai/grok-4.1-fast:free", schema=Plan
    )
    response = cast(Plan, response)
    logger.info(f"Number of steps: {len(response.steps)}...")

    return {
        "is_related_to_context": True,
        "plan": response,
        "step_state": [],
        "current_step_index": 0,
    }


@traceable
async def retrieve_internal_docs_node(state: State) -> dict[str, Any]:
    """Retrieve internal documents node."""
    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]
    logger.info(
        f"🛢 Using Vector DB\nRetrieving documents for Step {current_step_idx}: {current_step.question}"
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
    logger.info(f"Re-written queries: {rewritten_queries}")
    logger.info(
        f"Selected retrieval method: {retriever_method.method};\nRationale: {retriever_method.rationale}"
    )

    # Retrieve documents based on the selected method
    retrieved_docs: list[Document] = await aretrieve_internal_documents(
        method=retriever_method.method,
        rewritten_queries=rewritten_queries,
        target_section=current_step.target_section,
        k=K,
    )
    step_state = StepState(
        step_index=current_step_idx,
        question=current_step.question,
        rewritten_queries=rewritten_queries,
        retrieved_documents=retrieved_docs,
        summary="",
    )
    # Update the state with retrieved docs and step_state
    return {
        "step_state": [step_state],
        "retrieved_documents": retrieved_docs,
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
    logger.info(f"🌐 WEB SEARCH\nRe-written queries: {rewritten_queries}")

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

    step_state = StepState(
        step_index=current_step_idx,
        question=current_step.question,
        rewritten_queries=rewritten_queries,
        retrieved_documents=retrieved_docs,
        summary="",
    )
    # Update the state with retrieved docs and step_state
    return {
        "step_state": [step_state],
        "retrieved_documents": retrieved_docs,
    }


@traceable
async def rerank_and_compress_node(state: State) -> dict[str, Any]:
    """Rerank documents and then synthesize final context.

    Parameters
    ----------
    state : State
        The current state containing retrieved documents and other info.

    Returns
    -------
    dict[str, Any]
        Updated state with reranked documents and synthesized context.
    """
    rerank_result = await rerank_retrieved_documents(state)
    # Update state with reranked documents
    updated_state = {**state, **rerank_result}  # type: ignore
    compression_result = await compression_documents(updated_state)  # type: ignore
    return {**rerank_result, **compression_result}


@traceable
async def summarization_node(state: State) -> dict[str, Any]:
    """Synthesize final context from reranked documents. This node is also responsible for moving to
    the next step in the multi-step plan.
    """

    synthesized_context: str = state["synthesized_context"]
    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]
    rewritten_queries = [
        step for step in state["step_state"] if current_step_idx == step["step_index"]
    ][0]["rewritten_queries"]

    # Format document
    context: str = f"<CONTEXT>{synthesized_context}</CONTEXT>"
    sys_msg: str = prompt_builder.summarization_prompt(question=current_step.question)
    logger.info(f"Summarizing for Step {current_step_idx}: {current_step.question}")
    response = await remote_llm.ainvoke([SystemMessage(sys_msg), HumanMessage(context)])

    new_step_state: StepState = StepState(
        step_index=current_step_idx,
        question=current_step.question,
        rewritten_queries=rewritten_queries,
        retrieved_documents=state["retrieved_documents"],
        summary=response.content,  # type: ignore
    )
    logger.info(
        f"⚠️ Number of steps completed: {current_step_idx + 1} | Num iterations: {state.get('num_iterations', 0) + 1}"
    )

    # Append the new step state to the existing list and increment current step index
    return {
        "step_state": state.get("step_state", []) + [new_step_state],
        "current_step_index": current_step_idx + 1,
        "synthesized_context": response.content,
        "num_iterations": state.get("num_iterations", 0) + 1,
    }


@traceable
async def final_answer_node(state: State) -> dict[str, Any]:
    """Generate the final answer with citations based on all collected evidence."""

    logger.info("--- ✅: Generating Final Answer with Citations ---")
    # Gather all the evidence we've collected from ALL steps.
    final_context: str = ""
    for i, step in enumerate(state["step_state"]):
        final_context += f"\n--- Findings from Research Step {i + 1} ---\n"
        # Include the source metadata (section or URL) for each document to enable citations.
        for doc in step["retrieved_documents"]:
            source: str = doc.metadata.get("source_doc") or doc.metadata.get("url")  # type: ignore
            final_context += f"Source: {source}\nContent: {doc.page_content}\n\n"

    prompt: str = prompt_builder.final_answer_prompt(
        question=state["original_question"]
    )
    context: str = f"<CONTEXT>{final_context}</CONTEXT>"

    final_answer = await remote_llm.ainvoke(
        [SystemMessage(prompt), HumanMessage(context)]
    )
    # Update the state with the final answer and reset num_iterations
    return {
        "final_answer": final_answer.content,
        "num_iterations": 0,  # Reset counter for next question
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
        "num_iterations": 0,  # Reset counter for next question
    }


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


async def should_continue_condition(
    state: State, max_reasoning_interations: int = 8
) -> NextAction:
    """Determine if the current step should be retried.

    Parameters
    ----------
    state : State
        The current state of the agent.
    max_reasoning_interations : int, optional
        The maximum number of reasoning iterations allowed, by default 8.

    Returns
    -------
    NextAction
        The next action to take (CONTINUE or FINISH).
    """
    print("--- Evaluating Multi Step Reasoning Policy ---")
    is_related_to_context: bool = state.get("is_related_to_context", True)
    current_step_idx: int = state["current_step_index"]
    num_iterations: int = state.get("num_iterations", 0)

    # Checks
    # If query does NOT relate to the topics, finish immediately
    if not is_related_to_context:
        logger.info(" -> Query not related to context. Finishing...")
        return NextAction.FINISH

    # Are all the steps completed?
    if state["plan"] and (current_step_idx >= len(state["plan"].steps)):
        logger.info(f" -> Plan complete. {num_iterations} iterations. Finishing...")
        return NextAction.FINISH

    # Is the max num of iterations exhausted?
    if num_iterations >= max_reasoning_interations:
        logger.info(
            f" -> Max iterations reached. {num_iterations} iterations. Finishing..."
        )
        return NextAction.FINISH

    # Last retrieval step failed to find any docs
    if state.get("reranked_documents") is not None and not state["reranked_documents"]:
        logger.info(
            "⚠️ -> Retrieval failed for the last step. Continuing with next step in plan."
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
