# BookWyrmClient

The synchronous client for the BookWyrm API. This documentation is automatically generated from the source code.

::: bookwyrm.client.BookWyrmClient

## Error Handling

The client raises specific exceptions for different error conditions:

```python
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

try:
    response = client.get_citations(request)
except BookWyrmAPIError as e:
    print(f"API Error: {e}")
    if e.status_code:
        print(f"Status Code: {e.status_code}")
except BookWyrmClientError as e:
    print(f"Client Error: {e}")
```

## Session Management

The client uses a `requests.Session` internally for connection pooling and cookie persistence. You can close the session manually:

```python
client.close()
```

Or use the context manager for automatic cleanup:

```python
with BookWyrmClient() as client:
    # Use client
    pass
# Session is automatically closed
```
