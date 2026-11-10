"""Errors whose message is safe to show visitors. The underlying cause is logged, never sent."""


class ServiceError(Exception):
    pass


class RetrievalError(ServiceError):
    pass


class ChatHistoryError(ServiceError):
    pass


GENERIC_MESSAGE = "Something went wrong on our side. Please try again in a moment."


def public_message(error: Exception) -> str:
    return str(error) if isinstance(error, ServiceError) else GENERIC_MESSAGE
