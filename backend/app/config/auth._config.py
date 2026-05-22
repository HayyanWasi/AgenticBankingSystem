import os
from dotenv import load_dotenv

# Force Python to find and read the .env file in your directory structure
load_dotenv()

# Now os.getenv will successfully pull the real key from the .env file
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-development-key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))