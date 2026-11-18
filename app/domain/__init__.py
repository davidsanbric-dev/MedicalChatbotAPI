"""The rules of the service, in plain Python: what a passage is and when it is relevant,
how urgent a visitor's situation is, and what a handoff to a human holds.

Nothing here does I/O or imports SQLAlchemy, LangChain or another `app` package, so the
rules can be read and tested on their own. Services orchestrate them; repositories store them.
"""
