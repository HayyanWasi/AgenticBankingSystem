import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from app.models import Base

# By default, use SQLite locally. In production (Render), we will set DATABASE_URL to a PostgreSQL connection string.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///bank_data.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Only use check_same_thread for SQLite
connect_args = {"timeout": 30}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    echo=True,
    connect_args=connect_args,
)

# Only apply SQLite pragmas if using SQLite
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        try:
            cursor.execute("PRAGMA busy_timeout=30000")
        except Exception:
            pass
        cursor.close()

def build_database():
    print("Reading blueprints from models.py...")
    Base.metadata.create_all(bind=engine)
    print("Database built successfully.")

if __name__ == "__main__":
    build_database()
