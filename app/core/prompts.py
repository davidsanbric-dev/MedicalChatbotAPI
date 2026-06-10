from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt_template = ChatPromptTemplate.from_messages([
    ("system", '''You are an expert and helpful AI assistant. Your task is to answer the user's question based on two sources of information:

    1.  The relevant context extracted from a document provided to you.
    2.  The conversation history you have had with the user.

    Respond clearly, concisely, and friendly. If the answer is not found in the context or history, kindly indicate that you do not have that information.

    Document context:
    ---
    {context}
    ---'''),
    MessagesPlaceholder(variable_name="conversation_history"),
    ("human", "{input}")
])

query_rephraser_prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="conversation_history"),
    ("user", "Given the previous conversation, generate a search query that is autonomous and can be understood without the history. The query must be about the last topic discussed. Do not answer the question, just rephrase it."),
    ("user", "{input}")
])