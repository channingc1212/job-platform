from dotenv import load_dotenv
import os

def load_env_variables():
    """Load environment variables from .env file"""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
