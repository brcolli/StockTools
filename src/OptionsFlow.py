from tda.auth import easy_client
from tda.client import Client
from tda.streaming import StreamClient
import asyncio
import pprint


class OptionsFlowManager:

    def __init__(self, watchlist, queue_size=1):

        self.account_id = '275356186'
        self.key = 'FA3ISKLEGYIFXQRSUJQCB93AKXFRGZUK'
        self.callback_url = 'https://localhost:8080'
        self.token_path = '../doc/token'
        self.watchlist = watchlist
        self.queue = asyncio.Queue(queue_size)  # For gathering client work

        # Setup stream
        try:
            self.client = easy_client(api_key=self.key,
                                      redirect_uri=self.callback_url,
                                      token_path=self.token_path)
        except FileNotFoundError:

            from selenium import webdriver
            from webdriver_manager.chrome import ChromeDriverManager

            with webdriver.Chrome(ChromeDriverManager().install()) as driver:
                self.client = easy_client(api_key=self.key,
                                          redirect_uri=self.callback_url,
                                          token_path=self.token_path,
                                          webdriver_func=driver)

        self.stream = StreamClient(self.client, account_id=self.account_id)

    async def start_stream(self):

        await self.stream.login()
        await self.stream.add_options_book_handler(self.handle_options_book())
        await self.stream.options_book_subs(self.watchlist)

        asyncio.ensure_future(self.handle_queue())

        while True:
            await self.stream.handle_message()

    async def handle_options_book(self, msg):

        if self.queue.full():
            await self.queue.get()
        await self.queue.put(msg)

    async def handle_queue(self):

        while True:
            msg = await self.queue.get()
            pprint.pprint(msg)


async def main():

    symbols = ['AAPL', 'TSLA']
    ofm = OptionsFlowManager(symbols)

    await ofm.start_stream()

if __name__ == '__main__':
    asyncio.run(main())
