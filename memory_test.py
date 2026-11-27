
from Agent import get_agent
from langchain_core.messages import HumanMessage
from Database import init_db

# testing longterm and short-term memory

init_db()

agent = get_agent("gemini-3.5-flash-lite")

config = {
    "configurable": {
        "thread_id": "test_thread_id"
    }
}

try:
    for message_chunk, metadata in agent.stream(
        {
            "messages": [
                HumanMessage(content="what is my name?")
            ]
        },
        config=config,
        stream_mode="messages"
    ):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
