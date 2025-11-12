"""Configuration for live documentation code block testing."""

import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def setup_live_environment():
    """Set up environment for live API testing."""
    # Ensure API key is available
    api_key = os.getenv("BOOKWYRM_API_KEY")
    if not api_key:
        pytest.skip("BOOKWYRM_API_KEY environment variable not set - skipping live docs tests")
    
    # Ensure required data files exist
    data_dir = Path(__file__).parent.parent / "data"
    required_files = [
        "SOA_2025_Final.pdf",
        "country-of-the-blind.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(str(data_dir / file))
    
    if missing_files:
        pytest.skip(f"Required data files missing: {', '.join(missing_files)}")
    
    # Set up environment
    os.environ["BOOKWYRM_API_KEY"] = api_key
    yield data_dir


@pytest.fixture(autouse=True)
def cleanup_generated_files():
    """Clean up files generated during documentation tests."""
    yield
    
    # Clean up generated files after each test
    data_dir = Path(__file__).parent.parent / "data"
    cleanup_patterns = [
        "*pages*.json",
        "*raw.txt", 
        "*mapping.json",
        "*phrases.jsonl",
        "*summary*.json",
        "*citations*.json",
        "*analysis*.json",
        "character_positions.json"
    ]
    
    for pattern in cleanup_patterns:
        for file in data_dir.glob(pattern):
            try:
                file.unlink()
            except FileNotFoundError:
                pass


def pytest_configure(config):
    """Configure pytest for documentation testing."""
    config.addinivalue_line(
        "markers", "docs: mark test as documentation code block test"
    )
    config.addinivalue_line(
        "markers", "live: mark test as requiring live API access"
    )


def pytest_collection_modifyitems(config, items):
    """Mark all documentation tests as live tests."""
    for item in items:
        if "client-guide.md" in str(item.fspath):
            item.add_marker(pytest.mark.live)
            item.add_marker(pytest.mark.docs)
