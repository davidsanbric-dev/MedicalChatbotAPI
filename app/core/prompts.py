"""The two prompts behind each chat turn: rephrase the question, then answer it from the passages."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
