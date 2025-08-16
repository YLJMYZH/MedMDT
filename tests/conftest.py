# tests/conftest.py
import os
import pytest
from medmdt.config.settings import Settings


@pytest.fixture
def settings():
    return Settings(paddleocr_token="test-token")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from medmdt.config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
