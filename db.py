import os
import ssl
from sqlalchemy import text
import json
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    select,
    desc,
    func,
    delete,
    Boolean,
    LargeBinary
)
# from sqlalchemy.pool import NullPool
from typing import Optional, List, Dict
from dotenv import load_dotenv
import hashlib

import secrets
from datetime import datetime, timedelta

load_dotenv()


# Database URL with asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

if "sslmode=" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split("?")[0]

# SSL context for Neon 
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

# # Neon-optimized engine
# engine = create_async_engine(
#     DATABASE_URL,
#     echo=False,   # set True for debugging
#     future=True,
#     poolclass=NullPool,
#     connect_args={"ssl": ssl_context},
# )

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,

    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,

    connect_args={
        "ssl": ssl_context,
        "statement_cache_size": 0
    },
)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


# Models

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    hashed_password = Column(String(255), nullable=True)
    memory = Column(Text, nullable=True) 
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(300), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)

    user = relationship("User", back_populates="messages")
    conversation = relationship("Conversation", back_populates="messages")

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), nullable=False, index=True)
    token = Column(String(6), nullable=False)  # 6-digit code
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)  # This uses sqlalchemy.Boolean
    created_at = Column(DateTime, default=datetime.utcnow)

#new

class UserFact(Base):
    __tablename__ = "user_facts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", backref="facts")


class AddToMemory(Base):
    __tablename__ = "add_to_memory"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", backref="memory_entries")


class ChatImage(Base):
    __tablename__ = "chat_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    filename = Column(String, nullable=False)
    content_type = Column(String)
    image_data = Column(LargeBinary, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref="images")
    conversation = relationship("Conversation", backref="images")

class MessageFeedback(Base):
    __tablename__ = "message_feedbacks"
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    feedback_type = Column(String(20), nullable=False)  # 'good', 'bad'
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    message = relationship("Message", backref="feedbacks")
    user = relationship("User", backref="feedbacks")


class DeletedChat(Base):
    __tablename__ = "deleted_chats"

    id = Column(Integer, primary_key=True, index=True)
    original_conversation_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    title = Column(String(300), nullable=True)
    messages = Column(JSONB, nullable=True)
    deleted_at = Column(DateTime, default=datetime.utcnow)


class DeletedAccount(Base):
    __tablename__ = "deleted_accounts"

    id = Column(Integer, primary_key=True, index=True)
    original_user_id = Column(Integer, nullable=False)
    email = Column(String(255), nullable=False)
    name = Column(String(200), nullable=True)
    hashed_password = Column(String(255), nullable=True)
    conversations = Column(JSONB, nullable=True)
    user_facts = Column(JSONB, nullable=True)
    memory = Column(Text, nullable=True)
    deleted_at = Column(DateTime, default=datetime.utcnow)



# Password hashing helpers

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    return hash_password(plain_password) == hashed_password



# Database functions

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_db_health() -> bool:
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(User).limit(1))
            _ = result.scalars().first()
        return True
    except Exception as e:
        print(f"⚠️ DB health check failed: {e}")
        return False


async def async_get_or_create_user(email: str, name: str) -> User:
    """Fetch user by email, or create one without password (used for first-time chats)."""
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        if user:
            return user

        new_user = User(email=email, name=name)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        return new_user
    


async def async_create_conversation(user_id: int, title: str = "New Chat") -> Optional[int]:
    async with async_session_maker() as session:
        new_conv = Conversation(
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
        )
        session.add(new_conv)
        await session.commit()
        await session.refresh(new_conv)
        return new_conv.id


async def async_save_message(user_id: int, role: str, content: str, conversation_id: Optional[int] = None) -> Message:
    """Save a message and return the saved Message object."""
    async with async_session_maker() as session:
        msg = Message(
            user_id=user_id,
            role=role,
            content=content,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow(),
        )
        session.add(msg)

        # If conversation exists and has no title (New Chat), optionally update timestamps
        if conversation_id:
            await session.flush()
        await session.commit()
        await session.refresh(msg)
        return msg


async def async_get_history(user_id: int, limit: int = 50):
    """Optimized: Get recent messages directly without loading all conversations first"""
    async with async_session_maker() as session:
        # Just get the messages - much faster
        stmt = (select(Message)
                .where(Message.user_id == user_id)
                .order_by(Message.timestamp.desc())
                .limit(limit))
        
        result = await session.execute(stmt)
        messages = result.scalars().all()
        
        return list(reversed(messages))  # Return chronological order


# delete chats 

async def async_delete_conversation(conversation_id: int) -> bool:
    """
    Delete a conversation and ALL its traces so the agent has zero access.
    Archives to deleted_chats before deletion.
    """
    async with async_session_maker() as session:
        try:
            # 1. Fetch conversation
            conv_result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conv = conv_result.scalars().first()
            if not conv:
                return False

            user_id = conv.user_id

            # 2. Fetch all messages for archiving
            msgs_result = await session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp)
            )
            msgs = msgs_result.scalars().all()

            # 3. Serialize messages to JSON
            messages_json = [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None
                }
                for m in msgs
            ]

            # 4. Archive to deleted_chats BEFORE deleting
            await session.execute(
                text("""
                    INSERT INTO deleted_chats 
                        (original_conversation_id, user_id, title, messages)
                    VALUES 
                        (:conv_id, :user_id, :title, :messages)
                """),
                {
                    "conv_id": conversation_id,
                    "user_id": user_id,
                    "title": conv.title,
                    "messages": json.dumps(messages_json)
                }
            )

            # 5. Get message IDs for this conversation (needed for cleanup)
            msg_ids_result = await session.execute(
                text("SELECT id FROM messages WHERE conversation_id = :cid"),
                {"cid": conversation_id}
            )
            msg_ids = [row[0] for row in msg_ids_result.fetchall()]

            # 6. Delete message feedbacks
            await session.execute(
                text("""
                    DELETE FROM message_feedbacks 
                    WHERE message_id IN (
                        SELECT id FROM messages WHERE conversation_id = :cid
                    )
                """),
                {"cid": conversation_id}
            )

            # 7. Delete chat images
            await session.execute(
                text("DELETE FROM chat_images WHERE conversation_id = :cid"),
                {"cid": conversation_id}
            )

            # 8. Delete messages
            await session.execute(
                text("DELETE FROM messages WHERE conversation_id = :cid"),
                {"cid": conversation_id}
            )

            # 9. Delete conversation
            await session.execute(
                text("DELETE FROM conversations WHERE id = :cid"),
                {"cid": conversation_id}
            )

            # 10. ── CRITICAL: Clean add_to_memory entries that came from this conversation ──
            # We identify them by matching content against the archived messages
            if messages_json:
                deleted_contents = [m["content"] for m in messages_json]
                for content in deleted_contents:
                    await session.execute(
                        text("""
                            DELETE FROM add_to_memory 
                            WHERE user_id = :uid 
                            AND content = :content
                        """),
                        {"uid": user_id, "content": content}
                    )

            # 11. ── CRITICAL: Rebuild user memory excluding deleted conversation ──
            # Fetch remaining messages from OTHER conversations only
            remaining_msgs_result = await session.execute(
                text("""
                    SELECT m.role, m.content 
                    FROM messages m
                    WHERE m.user_id = :uid
                    ORDER BY m.timestamp DESC
                    LIMIT 100
                """),
                {"uid": user_id}
            )
            remaining_msgs = remaining_msgs_result.fetchall()

            if remaining_msgs:
                # Rebuild memory summary from remaining messages only
                lines = []
                for role, content in reversed(remaining_msgs):
                    prefix = "User" if role == "user" else "Assistant"
                    lines.append(f"{prefix}: {content[:200]}")
                new_memory_hint = "\n".join(lines[:20])  # Keep last 20 lines
            else:
                new_memory_hint = None  # No remaining conversations

            # Update user.memory to remove deleted chat context
            await session.execute(
                text("""
                    UPDATE users 
                    SET memory = :memory 
                    WHERE id = :uid
                """),
                {
                    "uid": user_id,
                    "memory": new_memory_hint  # NULL if no remaining chats
                }
            )

            await session.commit()
            print(f"✅ Conversation {conversation_id} fully deleted and archived. Memory rebuilt.")
            return True

        except Exception as e:
            await session.rollback()
            print(f"❌ async_delete_conversation error: {e}")
            return False


async def async_get_conversations_for_user(user_id: int) -> List[dict]:
    """Optimized with composite index"""
    async with async_session_maker() as session:
        # This query now uses idx_conversations_user_created index
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.created_at))
            .limit(50)  # Add limit to avoid loading hundreds of conversations
        )
        convs = result.scalars().all()

        conversations = []
        for conv in convs:
            # This query now uses idx_messages_conversation_timestamp index
            stmt = select(func.count(Message.id), func.max(Message.timestamp)).where(
                Message.conversation_id == conv.id
            )
            r = await session.execute(stmt)
            count, last_ts = r.first() or (0, None)

            conversations.append({
                "id": conv.id,
                "title": conv.title or "New Chat",
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "message_count": int(count or 0),
                "last_message_at": last_ts.isoformat() if last_ts else None,
            })
        return conversations



async def async_get_messages_for_conversation(conversation_id: int, limit: int = 50) -> List[Message]:
    """Get messages with limit to avoid loading huge conversations"""
    async with async_session_maker() as session:
        stmt = (select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.timestamp.desc())
                .limit(limit))
        result = await session.execute(stmt)
        messages = list(result.scalars().all())
        return list(reversed(messages))
    

async def async_create_password_reset_token(email: str) -> Optional[str]:
    """Create a password reset token and return it"""
    async with async_session_maker() as session:
        # Delete any existing tokens for this email
        await session.execute(
            delete(PasswordResetToken).where(PasswordResetToken.email == email)
        )
        
        # Generate 6-digit code
        token = ''.join(secrets.choice('0123456789') for _ in range(6))
        expires_at = datetime.utcnow() + timedelta(minutes=15)  # 15 minutes expiry
        
        reset_token = PasswordResetToken(
            email=email,
            token=token,
            expires_at=expires_at
        )
        
        session.add(reset_token)
        await session.commit()
        return token

async def async_verify_password_reset_token(email: str, token: str) -> bool:
    """Verify if a password reset token is valid"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.email == email,
                PasswordResetToken.token == token,
                PasswordResetToken.used == False,
                PasswordResetToken.expires_at > datetime.utcnow()
            )
        )
        reset_token = result.scalars().first()
        return reset_token is not None

async def async_use_password_reset_token(email: str, token: str) -> bool:
    """Mark a password reset token as used"""
    async with async_session_maker() as session:
        result = await session.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.email == email,
                PasswordResetToken.token == token,
                PasswordResetToken.used == False
            )
        )
        reset_token = result.scalars().first()
        
        if reset_token:
            reset_token.used = True
            await session.commit()
            return True
        return False

async def async_update_user_password(email: str, new_password: str) -> bool:
    """Update user's password"""
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        
        if user:
            user.hashed_password = hash_password(new_password)
            await session.commit()
            return True
        return False


async def save_user_fact(user_id: int, key: str, value: str):
    async with async_session_maker() as session:
        # Check if fact exists
        result = await session.execute(
            select(UserFact).where(UserFact.user_id == user_id, UserFact.key == key)
        )
        fact = result.scalars().first()
        if fact:
            fact.value = value
            fact.updated_at = datetime.utcnow()
        else:
            session.add(UserFact(user_id=user_id, key=key, value=value))
        await session.commit()

async def get_user_facts(user_id: int) -> Dict[str, str]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(UserFact.key, UserFact.value).where(UserFact.user_id == user_id)
        )
        return {key: value for key, value in result.all()}

async def add_memory_entry(user_id: int, content: str):
    async with async_session_maker() as session:
        session.add(AddToMemory(user_id=user_id, content=content))
        await session.commit()

        # Prune to last 15
        result = await session.execute(
            select(AddToMemory.id).where(AddToMemory.user_id == user_id).order_by(desc(AddToMemory.created_at))
        )
        ids = [r for r in result.scalars().all()]
        if len(ids) > 15:
            await session.execute(
                delete(AddToMemory).where(AddToMemory.id.not_in(ids[:6]))
            )
            await session.commit()

async def get_recent_memory(user_id: int) -> str:
    async with async_session_maker() as session:
        result = await session.execute(
            select(AddToMemory.content).where(AddToMemory.user_id == user_id).order_by(desc(AddToMemory.created_at)).limit(15)
        )
        return "\n".join(reversed(result.scalars().all()))


async def save_chat_image(user_id: int, conversation_id: int, image_data: bytes, filename: str = "uploaded.jpg", content_type: str = "image/jpeg"):
    async with async_session_maker() as session:
        session.add(ChatImage(
            user_id=user_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type=content_type,
            image_data=image_data
        ))
        await session.commit()


async def async_save_message_feedback(
    user_id: int, 
    message_id: int, 
    feedback_type: str, 
    comment: Optional[str] = None
):
    """Save or update message feedback"""
    async with async_session_maker() as session:
        # Check if feedback exists
        result = await session.execute(
            select(MessageFeedback).where(
                MessageFeedback.message_id == message_id,
                MessageFeedback.user_id == user_id
            )
        )
        existing = result.scalars().first()
        
        if existing:
            existing.feedback_type = feedback_type
            existing.comment = comment
            existing.created_at = datetime.utcnow()
        else:
            feedback = MessageFeedback(
                user_id=user_id,
                message_id=message_id,
                feedback_type=feedback_type,
                comment=comment
            )
            session.add(feedback)
        
        await session.commit()
        return True

#delete accounts.
async def async_delete_account(user_id: int) -> bool:
    """
    Permanently delete a user account.
    Archives everything to deleted_accounts table first.
    Agent has zero access to deleted_accounts.
    """
    async with async_session_maker() as session:
        try:
            # 1. Fetch user
            user_result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalars().first()
            if not user:
                return False

            # 2. Fetch all conversations with their messages
            convs_result = await session.execute(
                select(Conversation).where(Conversation.user_id == user_id)
            )
            convs = convs_result.scalars().all()

            conversations_json = []
            for conv in convs:
                msgs_result = await session.execute(
                    select(Message).where(Message.conversation_id == conv.id)
                    .order_by(Message.timestamp)
                )
                msgs = msgs_result.scalars().all()
                conversations_json.append({
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "messages": [
                        {
                            "role": m.role,
                            "content": m.content,
                            "timestamp": m.timestamp.isoformat() if m.timestamp else None
                        }
                        for m in msgs
                    ]
                })

            # 3. Fetch user facts
            facts_result = await session.execute(
                select(UserFact).where(UserFact.user_id == user_id)
            )
            facts = facts_result.scalars().all()
            facts_json = {f.key: f.value for f in facts}

            # 4. Archive to deleted_accounts
            await session.execute(
                text("""
                    INSERT INTO deleted_accounts
                        (original_user_id, email, name, hashed_password, conversations, user_facts, memory)
                    VALUES
                        (:uid, :email, :name, :hashed_password, :conversations, :user_facts, :memory)
                """),
                {
                    "uid": user_id,
                    "email": user.email,
                    "name": user.name,
                    "hashed_password": getattr(user, "hashed_password", None),
                    "conversations": json.dumps(conversations_json),
                    "user_facts": json.dumps(facts_json),
                    "memory": getattr(user, "memory", None),
                }
            )

            # 5. Delete all user data in order (foreign keys)
            await session.execute(
                text("""
                    DELETE FROM message_feedbacks WHERE user_id = :uid
                """), {"uid": user_id}
            )
            await session.execute(
                text("""
                    DELETE FROM message_feedbacks WHERE message_id IN (
                        SELECT id FROM messages WHERE user_id = :uid
                    )
                """), {"uid": user_id}
            )
            await session.execute(
                text("DELETE FROM chat_images WHERE user_id = :uid"),
                {"uid": user_id}
            )
            await session.execute(
                text("DELETE FROM add_to_memory WHERE user_id = :uid"),
                {"uid": user_id}
            )
            await session.execute(
                text("DELETE FROM user_facts WHERE user_id = :uid"),
                {"uid": user_id}
            )
            await session.execute(
                text("DELETE FROM password_reset_tokens WHERE email = :email"),
                {"email": user.email}
            )
            await session.execute(
                text("DELETE FROM messages WHERE user_id = :uid"),
                {"uid": user_id}
            )
            await session.execute(
                text("DELETE FROM conversations WHERE user_id = :uid"),
                {"uid": user_id}
            )
            await session.execute(
                text("DELETE FROM users WHERE id = :uid"),
                {"uid": user_id}
            )

            await session.commit()
            print(f"✅ User {user_id} ({user.email}) account deleted and archived")
            return True

        except Exception as e:
            await session.rollback()
            print(f"❌ async_delete_account error: {e}")
            return False