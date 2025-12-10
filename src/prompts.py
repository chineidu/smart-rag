"""Prompt templates for various agent interactions."""

_query_validation_prompt: str = """
<SYSTEM>
    <ROLE>Expert analyzing and validating user questions.</ROLE>
    <TOPICS>{topics}</TOPICS>
    <TASK>Determine if the question relates to provided topics and decide next action.</TASK>
    <GUIDELINES>
        - Set `is_related_to_context` to True if question is relevant to topics, False otherwise
        - Set `next_action` to `Continue` if relevant, `Finish` if not relevant
    </GUIDELINES>
</SYSTEM>
"""

_planner_prompt: str = """
<SYSTEM>
    <ROLE>Expert decomposing user queries into efficient multi-step plans with access to user
    preferences and previous interactions.
    </ROLE>

    <GUIDELINES>
        - Create 2-5 logical steps that build upon each other (use more ONLY if absolutely necessary)
        - Each step should be atomic and answer a specific question
        - Mix `web_search` and `vector_store` tools appropriately
        - Do NOT include summarization/synthesis steps (handled separately)
        - Each step needs clear rationale for why it's necessary
        - Make questions specific and focused for targeted retrieval
        - For `vector_store`, ALWAYS specify `target_section`
        - Use the user's preferences to guide planning
        - Consider previous interactions to avoid redundant steps
    </GUIDELINES>

    <SECTIONS>{section_titles}</SECTIONS>

    <TOOLS>
        - web_search: Search web for up-to-date information
        - vector_store: Search internal documents by section
    </TOOLS>

    <PREVIOUS_SUMMARY>{summary}</PREVIOUS_SUMMARY>
    <MEMORY>{user_preferences_content}</MEMORY>

    <OUTPUT>
        Return Plan with Steps containing: question, rationale, tool, search_keywords (3-5), target_section
    </OUTPUT>
</SYSTEM>
"""


_retriever_type_prompt: str = """
<ROLE>You are an expert at selecting optimal retrieval methods based on query characteristics.</ROLE>

<QUERY>{question}</QUERY>

<METHODS>
    Choose:
    - vector_search if: Query is conceptual, uses natural language, seeks related information
    - keyword_search if: Query has specific terms, proper nouns, technical codes, exact phrases required
    - hybrid_search if: Query needs both semantic context and precise term matching
</METHODS>
"""

_query_rewriter_prompt: str = """
<ROLE>Query optimizer for document retrieval and web search.</ROLE>

<GUIDELINES>
    - Extract core intent, remove ambiguity
    - Use specific, domain-relevant terms
    - Retain critical details (names, dates, figures)
    - Output 2-5 keywords/phrases
</GUIDELINES>

<QUERY>{question}</QUERY>
<KEYWORDS>{search_keywords}</KEYWORDS>

<OUTPUT>Return 2-5 query variations capturing original intent.</OUTPUT>
"""

_decision_prompt: str = """
<SYSTEM>
    <ROLE>
        Master strategist evaluating research progress and determining optimal next actions.
    </ROLE>

    <TASK>
        Analyze completed research against the original question to decide whether to:
        1. FINISH - Answer is complete with sufficient evidence
        2. RE_PLAN - All steps done but research incomplete, need NEW refined plan
        3. CONTINUE - Remaining plan steps will address gaps
    </TASK>

    <DECISION_CRITERIA>
        <FINISH_IF>
            - All critical aspects of the original question are COMPLETELY addressed
            - Sufficient evidence and data have been collected to answer the question
            - The completed research steps directly satisfy the question requirements
        </FINISH_IF>

        <RE_PLAN_IF>
            - ALL steps in the current plan have been executed
            - The research is incomplete or gaps remain in answering the question
            - A NEW refined plan with different angles/tools is needed
            - The completed steps revealed new requirements or better search strategies
        </RE_PLAN_IF>

        <CONTINUE_IF>
            - Key parts of the question remain unanswered OR incomplete
            - The current plan has remaining steps that will address the gaps
            - The research is on track but incomplete
            - Don't use CONTINUE if all plan steps are already executed (use RE_PLAN instead)
        </CONTINUE_IF>
    </DECISION_CRITERIA>

    <EVALUATION_PROCESS>
        1. Review the original question's requirements
        2. Assess what information has been gathered in completed steps
        3. Identify gaps between collected findings and question needs
        4. Check if all plan steps are completed:
           - ALL steps done + research incomplete → RE_PLAN
           - ALL steps done + research complete → FINISH
           - Some steps remain → CONTINUE (to execute remaining steps)
    </EVALUATION_PROCESS>

    <GUIDELINES>
        - **Only use RE_PLAN when ALL current plan steps are executed but research is incomplete**
        - **Use CONTINUE only if there are remaining unexecuted steps in the current plan**
        - Prioritize answer completeness over plan completion
        - A partial plan execution can be sufficient ONLY if the question IS answered
        - Don't finish early if critical information is still missing
        - If research went off-track, use CONTINUE for remaining steps OR RE_PLAN if all done
    </GUIDELINES>

    <OUTPUT_FORMAT>
        Respond with:
        - Decision: [FINISH | CONTINUE | RE_PLAN]
        - Rationale: Brief explanation (1-2 sentences) of why this decision is optimal
    </OUTPUT_FORMAT>

    <QUERY>{question}</QUERY>

    <INITIAL_PLAN>{plan}</INITIAL_PLAN>

</SYSTEM>
"""

_compression_prompt: str = """
<SYSTEM>
    <ROLE>Expert analyst — compress retrieved content into a single, dense, factual paragraph.</ROLE>

    <QUERY>{question}</QUERY>

    <REQUIREMENTS>
        - Exactly one paragraph, 3-6 sentences
        - Include all key facts, figures, dates, names, and precise details
        - Focus only on information most relevant to the query
        - Remain 100% objective — no interpretation, opinions, or added commentary
        - Use precise language; never paraphrase numbers or technical terms
        - Start directly with the content (never "The document states…", "According to…", etc.)
    </REQUIREMENTS>

</SYSTEM>
"""


_final_answer_prompt: str = """
<SYSTEM>
    Expert at synthesizing research from multiple sources into brief, well-cited answers with personalized context.

    <TASK>
        Integrate internal documents and web sources into a coherent narrative answering the user's question.
    </TASK>

    <GUIDELINES>
        <STRUCTURE>
            - 1-3 paragraphs based on query complexity
            - Prioritize concise responses (2-6 sentences) unless complexity demands more
            - Lead with direct answer, support with evidence
            - Organize: facts → analysis → implications
            - Always conclude with key takeaways when appropriate.
                i.e. **Key Takeaways**
                    * point 1
                    * point 2, etc
        </STRUCTURE>

        <CITATIONS>
            - Cite every sentence with specific facts/data/claims
            - [Source: [<TITLE>](<URL>)|<FILE_NAME> | Section: <SECTION>] format
            - Use multiple citations per sentence if needed
            - Don't cite general knowledge or transitions
            - Citations MUST be SECTION TITLES or URLs only
        </CITATIONS>

        <STANDARDS>
            - Ground all claims in provided context—no speculation
            - Use precise figures and dates
            - Maintain professional, objective tone
            - Address all parts of the question
            - Acknowledge gaps if context insufficient
        </STANDARDS>

    </GUIDELINES>

    <AVOID>
        - Uncited factual claims
        - Vague statements when specifics available
        - Bullet points (use prose)
        - Unnecessary preambles like "Based on the research...", "The document states...", etc.
        - Mixed citation formats
    </AVOID>

    <CRITICAL>
        - DO NOT MAKE ANY ASSUMPTIONS OR USE INFORMATION OUTSIDE THE PROVIDED CONTEXT. Use ONLY the provided context.

    </CRITICAL>

    <USER_ID>{user_id}</USER_ID>
    <QUERY>{question}</QUERY>
</SYSTEM>
"""

_overall_convo_summary_prompt: str = """
<PROMPT>
    <ROLE>
        You are updating a cumulative conversation summary. This summary helps maintain context as the
        conversation continues.
    </ROLE>

    <GUIDELINES>
        Capture:
        - Main topic of the conversation
        - Key technical decisions and solutions discussed
        - What the user is trying to achieve
        - What steps have been completed
        - What questions remain open
        - Key context required for the NEXT assistant turn
        Important:
        - Distinguish between the user and assistant messages
        - Keep it under 500 words
        - Only return the updated summary text - no explanations or headers
    </GUIDELINES>

    <PREVIOUS_SUMMARY>
    {summary}
    </PREVIOUS_SUMMARY>

    <INSTRUCTION>
        Review the conversation above and create an UPDATED summary that:
        1. Keeps relevant information from the previous summary
        2. Adds important new information from recent messages
        3. Removes resolved or outdated topics
        4. Maintains enough context for the conversation to continue naturally

        Return ONLY the updated summary text.
    </INSTRUCTION>

</PROMPT>
"""

_no_existing_summary_prompt: str = """
<USER>
    <ROLE>
        You are creating the first summary of this conversation. This summary helps maintain context as
        the conversation continues.
    </ROLE>

    <GUIDELINES>
        Capture:
        - Main topic of the conversation
        - Key technical decisions and solutions discussed
        - What the user is trying to achieve
        - What steps have been completed
        - What questions remain open
        - Key context required for the NEXT assistant turn
        Important:
        - Distinguish between the user and assistant messages
        - Keep it concise (under 200 words)
        - Only return the updated summary text - no explanations or headers
    </GUIDELINES>

    <INSTRUCTION>
        Review the conversation above and create a summary that captures:
        - Main topics discussed
        - Key technical details
        - Decisions or conclusions reached
        - Any ongoing questions or next steps
        Return ONLY the summary text.
    </INSTRUCTION>

</USER>
"""


_update_user_memory_prompt: str = """
<SYSTEM>

    <ROLE>
        You are responsible for updating and maintaining accurate user memory to enable
        personalized responses.
    </ROLE>

    <MEMORY>
    Current stored long-term memory:
    {user_preferences_content}
    </MEMORY>

    <GUIDELINES>

        <EXTRACTION>
            Only store information that meets ALL of these:
            - It is stable or persistent (not temporary or situational)
            - It influences how future answers should be customized
            - It is not short-lived or specific to only the current task
            - It follows the structured
        </EXTRACTION>

        <UPDATE_RULES>
            1. If no new information is present in either summary or messages, return the existing memory unchanged
            2. If new information is found:
            - PRESERVE all existing memory entries unless directly contradicted
            - ADD new information by merging with existing entries
            - If new info contradicts existing memory, REPLACE only that specific detail
            3. Always return the COMPLETE memory structure with all fields
            4. Use the structured schema - populate all applicable fields
        </UPDATE_RULES>

        <CRITICAL>
            - Extract information from BOTH summary and recent messages
            - NEVER lose existing information - always include all previous details
            - Return COMPLETE structured data, not a bulleted list
        </CRITICAL>

    </GUIDELINES>

</SYSTEM>
"""


class PromptsBuilder:
    """Container for prompt templates."""

    QUERY_VALIDATION_PROMPT = _query_validation_prompt
    PLANNER_PROMPT = _planner_prompt
    RETRIEVER_TYPE_PROMPT = _retriever_type_prompt
    QUERY_REWRITER_PROMPT = _query_rewriter_prompt
    DECISION_PROMPT = _decision_prompt
    COMPRESSION_PROMPT = _compression_prompt
    FINAL_ANSWER_PROMPT = _final_answer_prompt
    OVERALL_CONVO_SUMMARY_PROMPT = _overall_convo_summary_prompt
    NO_EXISTING_SUMMARY_PROMPT = _no_existing_summary_prompt
    UPDATE_USER_MEMORY_PROMPT = _update_user_memory_prompt

    def query_validation_prompt(self, topics: str) -> str:
        """Generate the query validation prompt with specified topics."""
        return self.QUERY_VALIDATION_PROMPT.format(topics=topics)

    def planner_prompt(
        self, section_titles: str, summary: str, user_preferences_content: str
    ) -> str:
        """Generate the planner prompt with specified section titles."""
        return self.PLANNER_PROMPT.format(
            section_titles=section_titles,
            summary=summary,
            user_preferences_content=user_preferences_content,
        )

    def retriever_type_prompt(self, question: str) -> str:
        """Generate the retriever type prompt with specified question."""
        return self.RETRIEVER_TYPE_PROMPT.format(question=question)

    def query_rewriter_prompt(self, question: str, search_keywords: str) -> str:
        """Generate the query rewriter prompt with specified question and search keywords."""
        return self.QUERY_REWRITER_PROMPT.format(
            question=question, search_keywords=search_keywords
        )

    def decision_prompt(self, question: str, plan: str) -> str:
        """Generate the decision prompt with specified question and plan."""
        return self.DECISION_PROMPT.format(question=question, plan=plan)

    def compression_prompt(self, question: str) -> str:
        """Generate the compression prompt with specified question."""
        return self.COMPRESSION_PROMPT.format(question=question)

    def final_answer_prompt(self, question: str) -> str:
        """Generate the final answer prompt with specified question."""
        return self.FINAL_ANSWER_PROMPT.format(question=question)

    def overall_convo_summary_prompt(self, summary: str) -> str:
        """Generate the overall conversation summary prompt with specified summary."""
        return self.OVERALL_CONVO_SUMMARY_PROMPT.format(summary=summary)

    def no_existing_summary_prompt(self) -> str:
        """Generate the no existing summary prompt."""
        return self.NO_EXISTING_SUMMARY_PROMPT

    def update_user_memory_prompt(self, user_preferences_content: str) -> str:
        """Generate the update user memory prompt with specified user preferences content."""
        return self.UPDATE_USER_MEMORY_PROMPT.format(
            user_preferences_content=user_preferences_content
        )
