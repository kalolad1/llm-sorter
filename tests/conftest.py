import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def api_key() -> str:
    """Get API key from environment, fail if not available."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key is None:
        raise ValueError("OPENROUTER_API_KEY is not set")
    return key
