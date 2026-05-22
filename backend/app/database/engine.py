from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from app.models import Base

DATABASE_URL = "sqlite:///bank_data.db"

engine = create_engine(
    DATABASE_URL,
    echo=True,
    connect_args={
        "timeout": 30,
        "check_same_thread": False,
    },
)

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
