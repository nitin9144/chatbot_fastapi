import os
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from pinecone import EmbedQuery, Pinecone

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value.startswith("your_"):
        raise RuntimeError(f"Set a valid {name} in the repository .env file.")
    return value


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
pinecone = Pinecone(api_key=_required_env("PINECONE_API_KEY"))
index = pinecone.index(_required_env("PINECONE_INDEX_NAME"))
namespace = os.getenv("PINECONE_NAMESPACE", "")
text_field = os.getenv("PINECONE_TEXT_FIELD", "text")


def retrieve_documents(query: str) -> dict[str, object]:
    result = index.documents.search(
        namespace=namespace,
        score_by=[EmbedQuery(field=text_field, query=query)],
        top_k=4,
        include_fields=["*"],
    )
    hits = result.matches
    context = []
    sources = []
    for hit in hits:
        document = hit.to_dict()
        text = document.pop(text_field, None)
        if not isinstance(text, str):
            raise RuntimeError(
                f"Pinecone result is missing the configured text field "
                f"{text_field!r}; set PINECONE_TEXT_FIELD to the index's text field."
            )
        context.append(text)
        sources.append(document)
    return {
        "context": context,
        "sources": sources,
    }


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: NotRequired[list[str]]
    sources: NotRequired[list[dict[str, object]]]


def retrieve_node(state: ChatState) -> dict[str, object]:
    query = next(
        (
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage) and isinstance(message.content, str)
        ),
        None,
    )
    if query is None:
        raise ValueError("The latest user message must contain text to search.")

    return retrieve_documents(query)


def chat_node(state: ChatState):
    if "context" not in state:
        raise RuntimeError("Retrieval node did not provide document context.")
    context = "\n\n---\n\n".join(state["context"])
    system_message = SystemMessage(
        content=(
            "You are the assistant for Nitin Sharma's portfolio. Answer portfolio "
            "questions using the retrieved documents below. If the documents contain "
            "the answer, answer directly and do not ask which project the user means. "
            "Do not invent facts. If the documents do not contain the answer, say so "
            "briefly.\n\n"
            f"Retrieved portfolio documents:\n{context}"
        )
    )
    response = llm.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


graph = StateGraph(ChatState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "chat_node")

chatbot = graph.compile()

if __name__ == "__main__":
    response = chatbot.invoke(
        {"messages": [HumanMessage(content="who is the president of india")]}
    )
    print(response["messages"][-1].content)
    print("Retrieved documents:")
    sources = response.get("sources")
    if sources is None:
        raise RuntimeError("Retrieval node did not return document sources.")
    for source in sources:
        print(f"- {source.get('_id')} (score: {source.get('_score')})")