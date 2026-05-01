import os
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class TicketPrediction(BaseModel):
    # We place justification first to enforce Chain-of-Thought reasoning
    justification: str = Field(
        description="Concise explanation of the routing/answering decision. Step 1: Check if answer is in docs. Step 2: Check if issue is high-risk (billing, passwords). Decide whether to reply or escalate."
    )
    status: Literal["replied", "escalated"] = Field(
        description="Choose 'replied' if the docs contain the answer or if the query is totally irrelevant. Choose 'escalated' ONLY if docs don't have the answer or explicitly instruct to contact human support."
    )
    product_area: str = Field(
        description="Most relevant support category or domain area based on the ticket and docs."
    )
    response: str = Field(
        description="The user-facing answer. MUST be grounded ONLY in the provided corpus. If escalating, politely state that the issue is being escalated to human support."
    )
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"] = Field(
        description="The best-fit classification of the user's issue."
    )

def get_agent():
    """
    Initializes the Gemini LLM with structured output forcing.
    Returns a runnable chain that takes a dictionary containing 'issue', 'subject', 'company', and 'context'.
    """
    # Use Gemini 2.5 Pro for complex reasoning and accurate structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    structured_llm = llm.with_structured_output(TicketPrediction)
    
    system_prompt = """You are a highly capable Support Triage Agent for HackerRank, Claude, and Visa.
Your job is to read a support ticket and the provided documentation corpus, and output exactly 5 fields.

CRITICAL RULES:
1. Grounding: You MUST ONLY use the facts provided in the "Retrieved Documentation". Do NOT use outside knowledge.
2. Replying vs Escalation: 
   - If ANY of the retrieved documents contain a solution or self-service instructions for the user's issue (e.g., how to delete a conversation, how to reinvite, how to report lost cards), you MUST set status to 'replied' and summarize the steps.
   - If the user asks a multi-part question (e.g., "how to do X and Y"), synthesize the answers from all provided documentation. Do NOT escalate simply because the answer is split across multiple documents.
   - ALWAYS set status to 'escalated' if the user reports a system outage, site-wide downtime, or a technical bug (e.g., "site is down", "none of the pages accessible").
   - Set status to 'escalated' only if the documentation does NOT contain a solution, or explicitly instruct the user to contact human support.
3. Irrelevant Queries: If the user asks a completely irrelevant question (e.g., "actor in Iron Man"), set request_type to 'invalid' and set status to 'replied' with a polite message that you are a support bot.
4. Chain of Thought: Your `justification` field must logically explain why you are choosing the status.

Retrieved Documentation:
{context}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Company: {company}\nSubject: {subject}\nIssue: {issue}")
    ])
    
    # We return the pipeline
    return prompt | structured_llm
