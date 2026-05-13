from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 1. Define where your database file lives
SQLALCHEMY_DATABASE_URL = "sqlite:///./bank_data.db"

# 2. Create the engine that talks to SQLite
# check_same_thread=False is strictly required for SQLite in this architecture
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 3. Create the SessionLocal factory
# This is what db_tools.py will call to open a connection
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Create the Base class that your models.py uses
Base = declarative_base()