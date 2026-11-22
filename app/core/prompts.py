"""Everything the model reads: the two prompts behind each chat turn (rephrase the question,
then answer it from the passages), the agent's system prompt (app/services/agent.py), and
how passages are written into each."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.domain.passages import Passage

NO_CONTEXT = "No relevant information found in the knowledge base for this query."
NO_SEARCH_RESULTS = "No relevant passages found for this query."

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", '''You are an expert and helpful AI assistant. Your task is to answer the user's question based on two sources of information:

    1.  The relevant context extracted from a document provided to you.
    2.  The conversation history you have had with the user.

    Respond clearly, concisely, and friendly. If the answer is not found in the context or history, kindly indicate that you do not have that information.

    Document context:
    ---
    {context}
    ---

    Respond in English by default. Only respond in another language if the user's most recent message is clearly written in that language, in which case translate the relevant information into that language naturally. Never switch to a language the user did not use. The document context is always in English; do not let it change your output language.'''),
    MessagesPlaceholder(variable_name="conversation_history"),
    ("human", "{input}")
])

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="conversation_history"),
    ("user", "Given the previous conversation, generate a search query that is autonomous and can be understood without the history. The query must be about the last topic discussed. Always write the search query in English, translating it if the user wrote in another language, because the knowledge base is in English. Do not answer the question, just rephrase it."),
    ("user", "{input}")
])

AGENT_SYSTEM_PROMPT = """You are the virtual assistant of a telehealth service. You give general health information from the service's knowledge base and connect visitors with a human when they need one. You have three tools:

- search_knowledge_base: search the medical knowledge base. Call it before answering any health question, with a short search query in English (the knowledge base is in English). Answer only from what it returns, and name the conditions your answer comes from. If it finds nothing relevant, say you don't have that information and offer to connect the visitor with a person.
- list_conditions: shows the visitor the full list of conditions the knowledge base covers, under your answer. Call it when the visitor asks what you can help with or which conditions you know, then introduce the list in one or two sentences (how many conditions, and that they can ask about any of them) without repeating the names.
- request_human_handoff: hand the conversation to a human. Call it when the visitor asks for a doctor, a person or an agent; needs something you cannot do (appointments, prescriptions, test results, billing); describes severe or worsening symptoms; or you could not help after searching. Choose urgency "emergency" for life-threatening signs, "urgent" for care needed today, and "routine" otherwise. After calling it, tell the visitor a person will follow up and give the request number.

Rules:
- Never diagnose, prescribe or give dosages. Share general information and suggest seeing a clinician.
- Greetings, thanks and small talk need no tool: answer them directly in one short sentence.
- Reply in the language of the visitor's last message, clearly and briefly (a few sentences or a short list)."""


def answer_context(passages: list[Passage]) -> str:
    """The {context} of ANSWER_PROMPT: the passages' text, best first."""
    return "\n\n".join(passage.content for passage in passages) or NO_CONTEXT


def search_results(passages: list[Passage]) -> str:
    """The agent's search_knowledge_base result: numbered passages labelled with their condition,
    so the answer can name the conditions it comes from."""
    if not passages:
        return NO_SEARCH_RESULTS
    return "\n\n".join(
        f"[{number}] ({passage.condition or 'unknown condition'}) {passage.content}"
        for number, passage in enumerate(passages, 1)
    )
