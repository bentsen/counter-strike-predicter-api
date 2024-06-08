from pydantic import BaseModel


class Chatbot(BaseModel):
    description: str | None
    image: str | None
