from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from app.database.engine import engine

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

