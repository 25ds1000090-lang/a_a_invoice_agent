from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Part(StrictModel):
    mediaType: str
    data: dict[str, Any]


class Message(StrictModel):
    messageId: str = Field(min_length=1)
    role: Literal["ROLE_USER", "ROLE_AGENT"]
    parts: list[Part] = Field(min_length=1)
    taskId: str | None = None
    contextId: str | None = None


class SendConfiguration(StrictModel):
    returnImmediately: bool = False
    historyLength: int = Field(default=20, ge=0, le=100)
    acceptedOutputModes: list[str] = Field(default_factory=list)


class SendRequest(StrictModel):
    message: Message
    configuration: SendConfiguration | None = None


class Artifact(StrictModel):
    artifactId: str
    name: str
    parts: list[Part] = Field(min_length=1)


class TaskStatus(StrictModel):
    state: Literal[
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_COMPLETED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_FAILED",
    ]
    timestamp: str


class Task(StrictModel):
    id: str
    contextId: str
    status: TaskStatus
    history: list[Message]
    artifacts: list[Artifact]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SendResponse(StrictModel):
    task: Task


class TaskListResponse(StrictModel):
    tasks: list[Task]


class ErrorBody(StrictModel):
    code: str
    message: str
