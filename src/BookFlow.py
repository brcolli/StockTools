from tda.streaming import StreamClient
import asyncio
import pprint
import importlib


TCM = importlib.import_module('TdaClientManager').TdaClientManager


class OptionsFlowManager:

    def __init__(self, watchlist, queue_size=1):

        self.tcm = TCM()
        self.watchlist = watchlist
        self.queue = asyncio.Queue(queue_size)  # For gathering client work

        # Add handlers
        self.tcm.stream.add_options_book_handler(self.handle_options_book)

    async def start_stream(self):

        await self.tcm.stream.login()
        await self.tcm.stream.quality_of_service(StreamClient.QOSLevel.EXPRESS)
        await self.tcm.stream.options_book_subs(self.watchlist)

        asyncio.ensure_future(self.handle_queue())

        while True:
            await self.tcm.stream.handle_message()

    async def handle_options_book(self, msg):

        if self.queue.full():
            await self.queue.get()
        await self.queue.put(msg)

    async def handle_queue(self):

        while True:
            msg = await self.queue.get()
            pprint.pprint(msg)


async def main():

    symbols = ['TSLA']
    ofm = OptionsFlowManager(symbols)

    await ofm.start_stream()

if __name__ == '__main__':
    asyncio.run(main())
