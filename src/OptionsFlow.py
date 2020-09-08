from tda.auth import easy_client
from tda.client import Client
from tda.streaming import StreamClient
import json
import asyncio

class OptionsFlowManager:

    def __init__(self):

        self.key = 'FA3ISKLEGYIFXQRSUJQCB93AKXFRGZUK'
        self.callback_url = 'https://localhost'

        self.client = easy_client(api_key=self.key,
                                  redirect_uri=self.callback_url,
                                  token_path='/tmp/token.pickle')
        self.stream = StreamClient(self.client)

    @staticmethod
    def options_handler(msg):
        print(json.dumps(msg, indent=4))

    async def read_stream(self):

        await self.stream.login()
        #await self.stream.