from sqlalchemy import create_engine
from models import Base 

DATABASE_URL = "sqlite:///bank_data.db"
engine = create_engine(DATABASE_URL, echo=True)



def build_database():
    print("Reading blueprints from models.py...")
    Base.metadata.create_all(bind=engine)
    print("Database built successfully.")

if __name__ == "__main__":
    build_database()


