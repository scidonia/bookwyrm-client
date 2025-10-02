# AsyncBookWyrmClient

The asynchronous client for the BookWyrm API, providing full async/await support. This documentation is automatically generated from the source code.

::: bookwyrm.async_client.AsyncBookWyrmClient

## Error Handling

The async client raises the same exceptions as the synchronous client:

```python
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

async def example_with_error_handling():
    try:
        async with AsyncBookWyrmClient() as client:
            response = await client.get_citations(request)
    except BookWyrmAPIError as e:
        print(f"API Error: {e}")
        if e.status_code:
            print(f"Status Code: {e.status_code}")
    except BookWyrmClientError as e:
        print(f"Client Error: {e}")
```

## Session Management

The async client uses `httpx.AsyncClient` internally. You can close the client manually:

```python
await client.close()
```

Or use the async context manager for automatic cleanup:

```python
async with AsyncBookWyrmClient() as client:
    # Use client
    pass
# Client is automatically closed
```

## Concurrent Operations

You can run multiple async operations concurrently:

```python
import asyncio

async def concurrent_operations():
    async with AsyncBookWyrmClient() as client:
        # Run multiple operations concurrently
        tasks = [
            client.get_citations(request1),
            client.get_citations(request2),
            client.summarize(summarize_request)
        ]
        
        results = await asyncio.gather(*tasks)
        return results

results = asyncio.run(concurrent_operations())
```

## Streaming with asyncio

Handle multiple streams concurrently:

```python
async def handle_multiple_streams():
    async with AsyncBookWyrmClient() as client:
        async def handle_citations():
            async for response in client.stream_citations(citation_request):
                print(f"Citation: {response}")
        
        async def handle_summarization():
            async for response in client.stream_summarize(summarize_request):
                print(f"Summary progress: {response}")
        
        # Run both streams concurrently
        await asyncio.gather(
            handle_citations(),
            handle_summarization()
        )

asyncio.run(handle_multiple_streams())
```
