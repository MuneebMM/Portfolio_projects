import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
import redis
import config

engine = create_engine(config.POSTGRES_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)

CACHE_TTL = 3600  # 1 hour


class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String(500), nullable=False)
    report_content = Column(Text, nullable=False)
    status = Column(String(50), default="completed")
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create tables if they don't exist."""
    Base.metadata.create_all(engine)


def save_report(topic: str, report: str, status: str = "completed") -> int:
    """Save a completed research report to PostgreSQL. Returns the report ID."""
    session = SessionLocal()
    try:
        entry = ResearchReport(topic=topic, report_content=report, status=status)
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry.id
    finally:
        session.close()


def get_report(report_id: int) -> dict | None:
    """Retrieve a specific report by ID."""
    session = SessionLocal()
    try:
        entry = session.query(ResearchReport).filter_by(id=report_id).first()
        if not entry:
            return None
        return {
            "id": entry.id,
            "topic": entry.topic,
            "report_content": entry.report_content,
            "status": entry.status,
            "created_at": entry.created_at.isoformat(),
        }
    finally:
        session.close()


def list_reports() -> list[dict]:
    """List all saved research reports."""
    session = SessionLocal()
    try:
        entries = (
            session.query(ResearchReport)
            .order_by(ResearchReport.created_at.desc())
            .all()
        )
        return [
            {
                "id": e.id,
                "topic": e.topic,
                "status": e.status,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ]
    finally:
        session.close()


# Redis cache helpers
def cache_set(key: str, value: str, ttl: int = CACHE_TTL):
    """Cache a value in Redis with TTL."""
    redis_client.setex(key, ttl, value)


def cache_get(key: str) -> str | None:
    """Get a cached value from Redis."""
    return redis_client.get(key)
