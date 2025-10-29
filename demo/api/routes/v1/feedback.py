"""FastAPI endpoint for handling user feedback."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from demo import create_logger
from demo.api.utilities.utilities import Feedback

logger = create_logger(name="feedback_api")


class FeedbackRequest(BaseModel):
    """Feedback request model."""

    session_id: str = Field(..., description="Session/checkpoint ID")
    message_index: int = Field(
        ..., ge=0, description="Index of the message in conversation"
    )
    user_message: str = Field(default="", description="User's question/prompt")
    assistant_message: str = Field(..., description="Assistant's response")
    sources: list[str] = Field(default_factory=list, description="List of source URLs")
    feedback: Feedback = Field(
        default=Feedback.NEUTRAL,
        description="Feedback type: 'positive', 'negative', or null",
    )
    timestamp: str | None = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"),
        description="Timestamp (auto-generated if not provided)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "message_index": 1,
                "user_message": "What is LangGraph?",
                "assistant_message": "LangGraph is a framework...",
                "sources": ["https://example.com"],
                "feedback": "positive",
            }
        }


class FeedbackResponse(BaseModel):
    """Feedback response model."""

    success: bool
    message: str
    feedback_id: str | None = None


# Create router
router = APIRouter(prefix="", tags=["feedback"])


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(feedback_data: FeedbackRequest) -> FeedbackResponse:
    """
    Submit user feedback for a chat message.

    Args:
        feedback_data: Feedback data including session, messages, and rating

    Returns:
        FeedbackResponse with success status
    """
    try:
        # Validate feedback type if provided
        if feedback_data.feedback and feedback_data.feedback not in [
            Feedback.POSITIVE,
            Feedback.NEGATIVE,
            Feedback.NEUTRAL,
        ]:
            raise HTTPException(
                status_code=400,
                detail="Feedback must be 'positive', 'negative', or null",
            )

        # Add timestamp if not provided
        if not feedback_data.timestamp:
            feedback_data.timestamp = datetime.now().isoformat()

        # Here you would typically save to a database
        # Example with SQLAlchemy:
        """
        from sqlalchemy.orm import Session
        from your_models import Feedback

        db_feedback = Feedback(
            session_id=feedback_data.session_id,
            message_index=feedback_data.message_index,
            user_message=feedback_data.user_message,
            assistant_message=feedback_data.assistant_message,
            sources=json.dumps(feedback_data.sources),
            feedback_type=feedback_data.feedback,
            timestamp=feedback_data.timestamp
        )
        db.add(db_feedback)
        db.commit()
        feedback_id = str(db_feedback.id)
        """

        # For now, log to console/file
        import json

        feedback_log_path = "feedback_logs.jsonl"

        log_entry = {
            "session_id": feedback_data.session_id,
            "message_index": feedback_data.message_index,
            "user_message": feedback_data.user_message,
            "assistant_message": feedback_data.assistant_message,
            "sources": feedback_data.sources,
            "feedback": feedback_data.feedback,
            "timestamp": feedback_data.timestamp,
        }

        # Append to JSONL file
        with open(feedback_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(
            f"[FEEDBACK] Session: {feedback_data.session_id}, "
            f"Index: {feedback_data.message_index}, "
            f"Type: {feedback_data.feedback}"
        )

        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            feedback_id=f"{feedback_data.session_id}_{feedback_data.message_index}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to save feedback: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save feedback: {str(e)}",
        ) from e


# Add this router to your main FastAPI app:
# app.include_router(router)
