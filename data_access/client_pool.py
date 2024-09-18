import asyncio
from typing import Callable


class ClientPool:
    def __init__(
        self, client_factory: Callable, max_wrappers: int = 1, *args, **kwargs
    ):
        """
        Initializes a pool of client instances.

        Parameters:
        - client_factory (Callable): A factory function or constructor for creating new client instances.
        - max_wrappers (int): The maximum number of clients in the pool.
        - *args, **kwargs: Additional arguments to pass to the client_factory when creating new clients.
        """
        self.client_factory = client_factory
        self.max_wrappers = max_wrappers
        self.client_args = args
        self.client_kwargs = kwargs
        self.clients = asyncio.Queue(max_wrappers)
        self._initialize_clients()

    def _initialize_clients(self):
        for _ in range(self.max_wrappers):
            client = self.client_factory(*self.client_args, **self.client_kwargs)
            self.clients.put_nowait(client)

    async def get_wrapper(self):
        return await self.clients.get()

    async def release_wrapper(self, client):
        await self.clients.put(client)
