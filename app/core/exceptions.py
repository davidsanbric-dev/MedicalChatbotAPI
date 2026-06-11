class RetrievalError(Exception):
    pass


class ChatHistoryError(Exception):
    """Raised when reading from or writing to the chat history store fails."""
    pass