from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base 

load_dotenv()

database_url = os.getenv('DATABASE_URL')

engine = create_engine(database_url)

# Create the tables in the database
Base.metadata.create_all(engine)

# Set up the session
SessionLocal = sessionmaker(bind=engine)

def get_db_session():
    """
    Provides a new database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
