"""Global pytest configuration."""

import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "docs: mark test as documentation code block test"
    )
    config.addinivalue_line("markers", "live: mark test as requiring live API access")


def pytest_collection_modifyitems(config, items):
    """Auto-mark documentation tests and filter code blocks."""
    for item in items:
        # Mark all .md files as docs tests
        if str(item.fspath).endswith(".md"):
            item.add_marker(pytest.mark.docs)
            item.add_marker(pytest.mark.live)

            # Skip bash/shell code blocks - check for pytest-codeblocks attributes
            if hasattr(item, "obj") and hasattr(item.obj, "info"):
                # pytest-codeblocks stores language in info attribute
                if item.obj.info in ["bash", "shell", "sh"]:
                    item.add_marker(pytest.mark.skip(reason="Skipping shell commands"))


@pytest.fixture(autouse=True)
def setup_docs_environment(request):
    """Set up environment for documentation tests."""
    # Only run for .md files
    if not str(request.fspath).endswith(".md"):
        return

    # Check for API key
    api_key = os.getenv("BOOKWYRM_API_KEY")
    if not api_key:
        pytest.skip(
            "BOOKWYRM_API_KEY environment variable not set - skipping live docs tests"
        )

    # Check for required data files
    data_dir = Path(__file__).parent / "data"
    required_files = ["SOA_2025_Final.pdf", "country-of-the-blind.txt"]

    missing_files = [
        str(data_dir / f) for f in required_files if not (data_dir / f).exists()
    ]
    if missing_files:
        pytest.skip(f"Required data files missing: {', '.join(missing_files)}")

    os.environ["BOOKWYRM_API_KEY"] = api_key
    yield data_dir

    # Cleanup generated files
    cleanup_patterns = [
        "*pages*.json",
        "*raw.txt",
        "*mapping.json",
        "*phrases.jsonl",
        "*summary*.json",
        "*citations*.json",
        "*analysis*.json",
        "character_positions.json",
    ]

    for pattern in cleanup_patterns:
        for file in data_dir.glob(pattern):
            try:
                file.unlink()
            except FileNotFoundError:
                pass
