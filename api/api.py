from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from chatbot_backend import ChatState, chatbot
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-portfolio-domain.com",
        "https://www.your-portfolio-domain.com",
        "http://localhost:5173",
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_request: ChatRequest) -> ChatResponse:
    if chat_request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The last message must be from the user.",
        )

    messages: list[BaseMessage] = [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in chat_request.messages
    ]
    graph_input: ChatState = {"messages": messages}
    response = chatbot.invoke(graph_input)
    reply = response["messages"][-1].content
    if not isinstance(reply, str):
        raise HTTPException(
            status_code=502,
            detail="The chatbot returned an unsupported response format.",
        )

    return ChatResponse(reply=reply)