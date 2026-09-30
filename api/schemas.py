from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Mouse(BaseModel):
    x: List[float]
    y: List[float]
    t: List[float]


class Keystroke(BaseModel):
    dwell: List[float] = Field(default_factory=list)
    flight: List[float] = Field(default_factory=list)


class Clicks(BaseModel):
    n_clicks: int = 0
    inter_click_ms: List[float] = Field(default_factory=list)
    press_ms: List[float] = Field(default_factory=list)


class SessionMeta(BaseModel):
    total_ms: float = 0
    n_events: int = 0
    event_rate_hz: float = 0
    has_focus_events: bool = False
    has_scroll: bool = False
    webdriver_flag: bool = False


class SessionIn(BaseModel):
    mouse: Mouse
    keystroke: Keystroke = Field(default_factory=Keystroke)
    clicks: Clicks = Field(default_factory=Clicks)
    session: SessionMeta = Field(default_factory=SessionMeta)
    baseline: str = "xgboost"


class CollectIn(BaseModel):
    """One labelled trial submitted by the collection UI."""
    mouse: Mouse
    keystroke: Keystroke = Field(default_factory=Keystroke)
    clicks: Clicks = Field(default_factory=Clicks)
    session: SessionMeta = Field(default_factory=SessionMeta)
    # collection metadata
    participant: Optional[str] = None
    label: str = "human"          # 'human' or a bot family name
    task: Optional[str] = None
    consent: bool = False
    user_agent: Optional[str] = None


class CollectOut(BaseModel):
    id: str
    stored: bool
    label: str


class ScoreOut(BaseModel):
    risk: float
    action: str
    p_bot_deep: float
    p_bot_baseline: float
    baseline_model: str
    reasons: List[str]
    attention_peaks_path_fraction: List[float]
    request_id: str
