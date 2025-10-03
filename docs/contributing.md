# Contributing

Thank you for your interest in contributing to the BookWyrm client library! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Setting up the Development Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-org/bookwyrm-client.git
   cd bookwyrm-client
   ```

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

1. **Install dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

1. **Set up pre-commit hooks:**

   ```bash
   pre-commit install
   ```

### Environment Variables

Create a `.env` file in the project root:

```bash
BOOKWYRM_API_KEY=your-test-api-key
BOOKWYRM_API_URL=https://api.bookwyrm.ai:443
```

## Project Structure

```
bookwyrm-client/
├── bookwyrm/
│   ├── __init__.py
│   ├── client.py          # Synchronous client
│   ├── async_client.py    # Asynchronous client
│   ├── models.py          # Pydantic models
│   └── cli.py             # Command-line interface
├── tests/
│   ├── test_client.py
│   ├── test_async_client.py
│   ├── test_models.py
│   └── test_cli.py
├── docs/                  # Documentation
├── examples/              # Example scripts
├── pyproject.toml         # Project configuration
└── README.md
```

## Making Changes

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:

```bash
make lint
```

Format code:

```bash
make format
```

### Testing

We use pytest for testing. Tests are located in the `tests/` directory.

Run tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=bookwyrm --cov-report=html
```

Run specific test files:

```bash
pytest tests/test_client.py
```

### Adding New Features

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

1. **Write tests first** (TDD approach):

   ```python
   # tests/test_new_feature.py
   def test_new_feature():
       # Test your new feature
       pass
   ```

1. **Implement the feature:**

   - Add new models to `models.py` if needed
   - Add client methods to `client.py` and `async_client.py`
   - Add CLI commands to `cli.py` if applicable

1. **Update documentation:**

   - Add docstrings to all new functions/classes
   - Update relevant documentation files in `docs/`
   - Add examples if appropriate

1. **Test your changes:**

   ```bash
   pytest
   make lint
   ```

### Adding New Models

When adding new Pydantic models:

1. **Define the model in `models.py`:**

   ```python
   class NewModel(BaseModel):
       field1: str
       field2: Optional[int] = None
       
       @model_validator(mode="after")
       def validate_fields(self):
           # Add validation logic
           return self
   ```

1. **Add to `__init__.py`:**

   ```python
   from .models import NewModel

   __all__ = [
       # ... existing exports
       "NewModel",
   ]
   ```

1. **Write tests:**

   ```python
   def test_new_model_validation():
       model = NewModel(field1="test")
       assert model.field1 == "test"
   ```

### Adding CLI Commands

When adding new CLI commands:

1. **Add the command to `cli.py`:**

   ```python
   @app.command()
   def new_command(
       arg1: Annotated[str, typer.Argument(help="Description")],
       option1: Annotated[bool, typer.Option(help="Option description")] = False,
   ):
       """Command description."""
       # Implementation
   ```

1. **Add tests:**

   ```python
   def test_new_command():
       result = runner.invoke(app, ["new-command", "test-arg"])
       assert result.exit_code == 0
   ```

1. **Update CLI documentation:**

   - Add command documentation to `docs/cli.md`
   - Include examples and option descriptions

## Testing Guidelines

### Unit Tests

- Test all public methods and functions
- Test error conditions and edge cases
- Use mocking for external API calls
- Aim for high test coverage (>90%)

### Integration Tests

- Test end-to-end workflows
- Use real API calls with test data
- Mark slow tests with `@pytest.mark.slow`

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from bookwyrm import BookWyrmClient
from bookwyrm.models import CitationRequest

class TestBookWyrmClient:
    def test_get_citations_success(self):
        # Arrange
        client = BookWyrmClient(api_key="test-key")
        request = CitationRequest(...)
        
        # Act
        with patch('requests.Session.post') as mock_post:
            mock_post.return_value.json.return_value = {...}
            response = client.get_citations(request)
        
        # Assert
        assert response.total_citations == 1
        mock_post.assert_called_once()
    
    def test_get_citations_api_error(self):
        # Test error handling
        pass
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 0.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param1 is empty.
        BookWyrmAPIError: When API request fails.
        
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    return True
```

### Documentation Files

- Keep documentation up to date with code changes
- Use clear, concise language
- Include practical examples
- Test code examples to ensure they work

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass:**

   ```bash
   pytest
   make lint
   ```

1. **Update documentation** if needed

1. **Create a pull request:**

   - Use a descriptive title
   - Include a detailed description of changes
   - Reference any related issues
   - Add screenshots for UI changes

1. **Pull request template:**

   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Tests pass locally
   - [ ] Added tests for new functionality
   - [ ] Updated documentation

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   ```

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:

```
feat(client): add support for PDF extraction
fix(cli): handle missing API key gracefully
docs(api): update client documentation
```

## Release Process

### Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Creating a Release

1. **Update version in `pyproject.toml`**
1. **Update CHANGELOG.md**
1. **Create a git tag:**
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```
1. **GitHub Actions will automatically publish to PyPI**

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: maintainer@example.com for private matters

### Reporting Issues

When reporting bugs, include:

- Python version
- Library version
- Operating system
- Minimal code example
- Full error traceback
- Expected vs actual behavior

### Feature Requests

When requesting features:

- Describe the use case
- Explain why it's needed
- Provide examples of how it would be used
- Consider implementation complexity

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitHub contributor graphs

Thank you for contributing to BookWyrm! 🎉
