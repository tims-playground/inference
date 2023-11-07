import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

FrameTimestamp = datetime
FrameID = int


class UpdateSeverity(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass(frozen=True)
class StatusUpdate:
    timestamp: datetime
    severity: UpdateSeverity
    event_type: str
    payload: dict
