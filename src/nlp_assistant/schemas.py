"""
Pydantic schemas for the NLP & Gemini Multimodal Crowd Intelligence Platform.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import datetime


# ── Legacy / Basic NLP Schemas ──

class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Operator's natural language question")
    session_id: Optional[str] = Field(None, description="Optional session filter")


class ChatSource(BaseModel):
    source_type: str
    timestamp: str
    details: str


class ChatQueryResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    sources: List[ChatSource]
    data: Dict[str, Any]
    timestamp: str


# ── Gemini Conversational AI Schemas ──

class AIChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%H:%M:%S"))


class AIChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Operator prompt")
    session_id: str = Field(default="default_session", description="Unique conversation session ID")
    lang: str = Field(default="en", pattern="^(en|hi|mr)$", description="Target language (en, hi, mr)")
    clear_history: bool = Field(default=False, description="Clear history before this query")


class AIChatResponse(BaseModel):
    answer: str
    lang: str = "en"
    session_id: str
    sources: List[ChatSource] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[str] = Field(default_factory=list)
    fallback_used: bool = False
    timestamp: str


class ClearHistoryRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session ID to clear")


class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[AIChatMessage]


# ── Gemini Vision Analyst Schemas ──

class VisionAnalysisRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Optional base64-encoded image. If None, live frame is used.")
    camera_id: str = Field(default="CAM-1", description="Camera identifier")
    lang: str = Field(default="en", pattern="^(en|hi|mr)$", description="Response language")
    use_live_frame: bool = Field(default=True, description="Capture latest live frame if image_base64 is omitted")


class VisionAnalysisResponse(BaseModel):
    scene_summary: str
    observed_conditions: List[str] = Field(default_factory=list)
    potential_safety_concerns: List[str] = Field(default_factory=list)
    supporting_visual_evidence: List[str] = Field(default_factory=list)
    confidence_or_uncertainty: str
    recommended_operator_actions: List[str] = Field(default_factory=list)
    timestamp: str
    camera_id: str
    snapshot_url: Optional[str] = None
    fallback_used: bool = False


# ── AI Incident Explanation Schemas ──

class IncidentExplanationRequest(BaseModel):
    incident_id: Optional[str] = Field(None, description="Optional SQLite incident report ID")
    alert_payload: Optional[Dict[str, Any]] = Field(None, description="Structured metrics/alert payload")
    lang: str = Field(default="en", pattern="^(en|hi|mr)$")
    include_snapshot: bool = Field(default=True, description="Attach live camera snapshot to explanation")


class IncidentExplanationResponse(BaseModel):
    incident_id: Optional[str] = None
    alert_level: str
    official_risk_score: float
    causal_indicators: List[str] = Field(default_factory=list)
    gemini_explanation: str
    recommended_actions: List[str] = Field(default_factory=list)
    visual_evidence: Optional[str] = None
    timestamp: str
    fallback_used: bool = False


# ── Incident Report & Audit Schemas ──

class IncidentReportCreate(BaseModel):
    event_type: str = Field(..., description="Category of event (e.g. STAMPEDE_RISK, DENSITY_SURGE)")
    severity: str = Field(..., description="Severity level: WARNING, DANGER, CRITICAL")
    camera_id: str = Field(default="CAM-1", description="Camera or sensor identifier")
    zone: str = Field(default="Zone-1", description="Zone location")
    count: Optional[int] = Field(None, ge=0, description="Observed crowd headcount")
    density: Optional[float] = Field(None, ge=0.0, le=1.0, description="Measured density index")
    occupancy_pct: Optional[float] = Field(None, ge=0.0, description="Occupancy percentage")
    evidence_url: Optional[str] = Field(None, description="Video recording file or snapshot path")
    operator_notes: Optional[str] = Field(None, description="Initial operator observations")
    ai_explanation: Optional[str] = Field(None, description="Gemini-generated causal explanation")
    visual_observations: Optional[str] = Field(None, description="Gemini Vision scene observations")
    recommended_actions: Optional[List[str]] = Field(default_factory=list, description="Recommended operator actions")


class IncidentReport(BaseModel):
    id: str
    timestamp: float
    time_str: str
    event_type: str
    severity: str
    camera_id: str
    zone: str
    count: int
    density: float
    occupancy_pct: float
    risk_score: float
    evidence_url: Optional[str] = None
    verification_status: str = "UNVERIFIED"  # UNVERIFIED | VERIFIED_FALSE_POSITIVE | CONFIRMED_HAZARD | RESOLVED
    operator_notes: Optional[str] = None
    ai_explanation: Optional[str] = None
    visual_observations: Optional[str] = None
    recommended_actions: List[str] = Field(default_factory=list)
    verified_at: Optional[float] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class VerificationUpdate(BaseModel):
    status: str = Field(..., pattern="^(UNVERIFIED|VERIFIED_FALSE_POSITIVE|CONFIRMED_HAZARD|RESOLVED)$")
    operator_notes: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, description="Search query")
    limit: int = Field(default=10, ge=1, le=50)
    min_similarity: float = Field(default=0.05, ge=0.0, le=1.0)


class SemanticSearchItem(BaseModel):
    id: str
    event_type: str
    level: str
    message: str
    timestamp: float
    time_str: str
    similarity: float
    evidence_link: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class SemanticSearchResponse(BaseModel):
    query: str
    count: int
    results: List[SemanticSearchItem]


class TranslateAlertRequest(BaseModel):
    alerts: List[Dict[str, Any]] = Field(..., description="List of alert dictionaries")
    target_lang: str = Field(default="en", pattern="^(en|hi|mr)$")


class TranslateAlertResponse(BaseModel):
    target_lang: str
    alerts: List[Dict[str, Any]]
