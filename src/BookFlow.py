from tda.streaming import StreamClient
import asyncio
import pprint
import importlib
import json


TCM = importlib.import_module('TdaClientManager').TdaClientManager


class OptionsFlowManager:

    def __init__(self, watchlist, queue_size=1):

        self.tcm = TCM()
        self.watchlist = watchlist
        self.queue = asyncio.Queue(queue_size)  # For gathering client work

    def start_stream(self):
        asyncio.get_event_loop().run_until_complete(self.read_stream())

    async def read_stream(self):

        await self.tcm.stream.login()
        await self.tcm.stream.quality_of_service(StreamClient.QOSLevel.EXPRESS)
        await self.tcm.stream.options_book_subs(self.watchlist)

        # Add handlers
        self.tcm.stream.add_options_book_handler(self.handle_options_book)

        print('Starting stream for {}'.format(','.join(self.watchlist)))

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


def main():

    symbols = ['AMC']
    ofm = OptionsFlowManager(symbols)

    ofm.start_stream()


if __name__ == '__main__':
    main()
