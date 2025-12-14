"""
Database models for EduAssist application
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Meeting(Base):
    """
    Meeting model - represents a meeting session
    """
    __tablename__ = "meetings"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(50), default="pending")  # pending, in_progress, completed
    meeting_context = Column(Text, nullable=True)  # Meeting context generated from documents
    
    # Relationships
    transcript = relationship("Transcript", back_populates="meeting", uselist=False, cascade="all, delete-orphan")
    summarizes = relationship("Summarize", back_populates="meeting", cascade="all, delete-orphan")
    conversation = relationship("Conversation", back_populates="meeting", uselist=False, cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="meeting", cascade="all, delete-orphan")


class Transcript(Base):
    """
    Transcript model - stores the raw transcript of a meeting
    """
    __tablename__ = "transcripts"
    
    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), unique=True, nullable=False)
    content = Column(Text, nullable=False)
    language = Column(String(10), default="vi")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    duration_ms = Column(Integer, nullable=True)  # Duration in milliseconds
    word_count = Column(Integer, nullable=True)
    
    # Relationship
    meeting = relationship("Meeting", back_populates="transcript")


class Summarize(Base):
    """
    Summarize model - stores different types of summaries for a meeting
    """
    __tablename__ = "summarizes"
    
    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    summary_type = Column(String(50), default="general")  # general, detailed, key_points, action_items
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    meeting = relationship("Meeting", back_populates="summarizes")


class Document(Base):
    """
    Document model - stores uploaded documents for a meeting
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, txt, etc.
    file_size = Column(Integer, nullable=True)  # Size in bytes
    
    # Embedding information
    is_embedded = Column(Boolean, default=False)
    embedding_model = Column(String(100), nullable=True)
    vector_store_path = Column(String(512), nullable=True)
    chunk_count = Column(Integer, nullable=True)
    
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    meeting = relationship("Meeting", back_populates="documents")


class Conversation(Base):
    """
    Conversation model - represents a Q&A conversation within a meeting
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    meeting = relationship("Meeting", back_populates="conversation")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    """
    Message model - individual messages in a conversation
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # human, ai
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Optional metadata - using 'extra_data' to avoid SQLAlchemy reserved word
    extra_data = Column(JSON, nullable=True)  # Can store additional info like sources, confidence, etc.
    
    # Relationship
    conversation = relationship("Conversation", back_populates="messages")
