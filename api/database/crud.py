"""
CRUD operations for database models
"""
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import Meeting, Transcript, Summarize, Document, Conversation, Message


# ============================================================================
# MEETING CRUD
# ============================================================================

def create_meeting(db: Session, title: str, description: Optional[str] = None) -> Meeting:
    """Create a new meeting"""
    meeting = Meeting(
        title=title,
        description=description,
        status="pending"
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    
    # Create associated conversation
    conversation = Conversation(meeting_id=meeting.id)
    db.add(conversation)
    db.commit()
    
    return meeting


def get_meeting(db: Session, meeting_id: int) -> Optional[Meeting]:
    """Get meeting by ID"""
    return db.query(Meeting).filter(Meeting.id == meeting_id).first()


def get_all_meetings(db: Session, skip: int = 0, limit: int = 100) -> List[Meeting]:
    """Get all meetings with pagination"""
    return db.query(Meeting).offset(skip).limit(limit).all()


def update_meeting(db: Session, meeting_id: int, **kwargs) -> Optional[Meeting]:
    """Update meeting fields"""
    meeting = get_meeting(db, meeting_id)
    if not meeting:
        return None
    
    for key, value in kwargs.items():
        if hasattr(meeting, key):
            setattr(meeting, key, value)
    
    meeting.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(meeting)
    return meeting


def delete_meeting(db: Session, meeting_id: int) -> bool:
    """Delete a meeting and all related data"""
    meeting = get_meeting(db, meeting_id)
    if not meeting:
        return False
    
    db.delete(meeting)
    db.commit()
    return True


# ============================================================================
# TRANSCRIPT CRUD
# ============================================================================

def create_transcript(
    db: Session,
    meeting_id: int,
    content: str,
    duration_ms: Optional[int] = None,
    language: str = "vi"
) -> Transcript:
    """Create transcript for a meeting"""
    transcript = Transcript(
        meeting_id=meeting_id,
        content=content,
        duration_ms=duration_ms,
        language=language,
        word_count=len(content.split())
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return transcript


def get_transcript(db: Session, meeting_id: int) -> Optional[Transcript]:
    """Get transcript by meeting ID"""
    return db.query(Transcript).filter(Transcript.meeting_id == meeting_id).first()


def update_transcript(db: Session, meeting_id: int, content: str) -> Optional[Transcript]:
    """Update transcript content"""
    transcript = get_transcript(db, meeting_id)
    if not transcript:
        return None
    
    transcript.content = content
    transcript.word_count = len(content.split())
    transcript.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(transcript)
    return transcript


# ============================================================================
# SUMMARIZE CRUD
# ============================================================================

def create_summarize(
    db: Session,
    meeting_id: int,
    content: str,
    summary_type: str = "general",
    title: Optional[str] = None
) -> Summarize:
    """Create a summary for a meeting"""
    summarize = Summarize(
        meeting_id=meeting_id,
        content=content,
        summary_type=summary_type,
        title=title
    )
    db.add(summarize)
    db.commit()
    db.refresh(summarize)
    return summarize


def get_summarizes(db: Session, meeting_id: int) -> List[Summarize]:
    """Get all summaries for a meeting"""
    return db.query(Summarize).filter(Summarize.meeting_id == meeting_id).all()


def get_summarize_by_type(db: Session, meeting_id: int, summary_type: str) -> Optional[Summarize]:
    """Get specific type of summary"""
    return db.query(Summarize).filter(
        Summarize.meeting_id == meeting_id,
        Summarize.summary_type == summary_type
    ).first()


# ============================================================================
# DOCUMENT CRUD
# ============================================================================

def create_document(
    db: Session,
    meeting_id: int,
    filename: str,
    file_path: str,
    file_type: str,
    file_size: Optional[int] = None
) -> Document:
    """Create a document record"""
    document = Document(
        meeting_id=meeting_id,
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        is_embedded=False
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def get_documents(db: Session, meeting_id: int) -> List[Document]:
    """Get all documents for a meeting"""
    return db.query(Document).filter(Document.meeting_id == meeting_id).all()


def update_document_embedding(
    db: Session,
    document_id: int,
    vector_store_path: str,
    embedding_model: str,
    chunk_count: int
) -> Optional[Document]:
    """Update document with embedding information"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return None
    
    document.is_embedded = True
    document.vector_store_path = vector_store_path
    document.embedding_model = embedding_model
    document.chunk_count = chunk_count
    db.commit()
    db.refresh(document)
    return document


def delete_document(db: Session, document_id: int) -> bool:
    """Delete a document"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return False
    
    db.delete(document)
    db.commit()
    return True


# ============================================================================
# CONVERSATION & MESSAGE CRUD
# ============================================================================

def get_conversation(db: Session, meeting_id: int) -> Optional[Conversation]:
    """Get conversation for a meeting"""
    return db.query(Conversation).filter(Conversation.meeting_id == meeting_id).first()


def add_message(
    db: Session,
    meeting_id: int,
    role: str,
    content: str,
    extra_data: Optional[Dict[str, Any]] = None
) -> Message:
    """Add a message to the conversation"""
    conversation = get_conversation(db, meeting_id)
    if not conversation:
        # Create conversation if it doesn't exist
        conversation = Conversation(meeting_id=meeting_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    
    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        extra_data=extra_data
    )
    db.add(message)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(message)
    return message


def get_messages(db: Session, meeting_id: int, limit: Optional[int] = None) -> List[Message]:
    """Get all messages in a conversation"""
    conversation = get_conversation(db, meeting_id)
    if not conversation:
        return []
    
    query = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.created_at)
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def get_conversation_history(db: Session, meeting_id: int, last_n: int = 10) -> List[Dict[str, str]]:
    """Get conversation history in format for LLM"""
    messages = get_messages(db, meeting_id)
    
    # Get last N messages
    recent_messages = messages[-last_n:] if len(messages) > last_n else messages
    
    history = []
    for msg in recent_messages:
        history.append({
            "role": msg.role,
            "content": msg.content
        })
    
    return history


def clear_conversation(db: Session, meeting_id: int) -> bool:
    """Clear all messages in a conversation"""
    conversation = get_conversation(db, meeting_id)
    if not conversation:
        return False
    
    db.query(Message).filter(Message.conversation_id == conversation.id).delete()
    db.commit()
    return True
