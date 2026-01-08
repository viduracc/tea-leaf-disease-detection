from sqlalchemy import create_engine, Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import os

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app_data", "tea_leaf.db")
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="owner")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    file_path = Column(String)
    heatmap_path = Column(String, nullable=True)
    predicted_class = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="predictions")


def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Get a database session."""
    return SessionLocal()
