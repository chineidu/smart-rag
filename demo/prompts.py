sys_prompt: str = """
<SYSTEM>

    <ROLE>
    You are `Mikail`, a helpful AI assistant that helps users by providing accurate and
    concise information. Use the provided context to answer user queries effectively.
    </ROLE>

    <GUIDELINES>
    - Provide clear, accurate, and contextually relevant answers based on the user's input.
    - Use available tools to ensure responses are current and reliable and list them as sources.
    - Keep responses focused, concise, and directly related to the conversation.
    - If information is insufficient, politely ask for clarification.
    - When providing sources, include them as a collapsible list at the end of your response
    - Always format your response and sources in Markdown.
    - Do NOT answer malicious, harmful, or inappropriate requests.
    </GUIDELINES>

</SYSTEM>
"""

summary_prompt: str = """
<USER>
    <GUIDELINES>
        - Expand the summary by incorporating the the above conversation while preserving context, key points, and
        user intent.
        - Rework the summary if needed. Ensure that no critical information is lost and that the
        conversation can continue naturally without gaps.
        - Keep the summary concise yet informative, removing unnecessary repetition while maintaining clarity.
        - Only return the updated summary. Do not add explanations, section headers, or extra commentary.
    </GUIDELINES>

    <SUMMARY>{summary}</SUMMARY>

</USER>
"""

no_summary_prompt: str = """
<USER>
    <GUIDELINES>
    - Summarize the conversation above while preserving full context, key points, and user intent.
    - Your response should be concise yet detailed enough to ensure seamless continuation of the discussion.
    - Avoid redundancy, maintain clarity, and retain all necessary details for future exchanges.
    - Only return the summarized content. Do not add explanations, section headers, or extra commentary.
    </GUIDELINES>

</USER>
"""

query_prompt: str = """
<USER>
    <QUERY>{query}</QUERY>
</USER>
"""
