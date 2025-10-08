import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from time import monotonic

import quart
from loguru import logger
from quart import Response, request

from tlserver.translator import Translator


@contextmanager
def timed(action):
    tick = monotonic()
    try:
        yield
    finally:
        tock = monotonic()
        logger.info(f"{action} in {tock - tick:.3f}s")


class Handler(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def receive_command(self) -> Response:
        pass


class LegacyTranslatorHandler(Handler):
    def __init__(self, translator: Translator) -> None:
        self.translator = translator
        self.translator.activate()

    async def receive_command(self) -> Response:
        with logger.contextualize(uid=uuid.uuid4()), timed("command handled") as _:
            data: dict = await request.get_json(force=True)
            message: str = data.get("message")
            content = data.get("content")

            logger.info(f"received command '{message}'  |  {content}")

            # special case for legacy support
            if message == "translate sentences" and isinstance(content, list):
                logger.debug(
                    "legacy support: translate single sentence converted to batch translate"
                )
                message = "translate batch"

            response = None

            match message:
                case "close server":
                    # TODO: implement "ending the handler" rather than the server
                    # we'd like to stop handling the port too if possible
                    # (how do we stop only one app?)
                    # for now we ignore this
                    logger.debug("die command ignored")

                case "check if server is ready":
                    response = self.translator.is_ready

                case "translate sentences":
                    if isinstance(content, str):
                        with timed("translated") as _:
                            translation = await self.translator.translate(content)
                        response = translation

                case "translate batch":
                    if isinstance(content, list):
                        with timed("batch translated") as _:
                            translation = await self.translator.translate_batch(content)
                        response = translation

                case "change input language":
                    if isinstance(content, str):
                        response = self.translator.change_input_language(content)

                case "change output language":
                    if isinstance(content, str):
                        response = self.translator.change_output_language(content)

                case "pause":
                    response = self.translator.pause()

                case "resume":
                    response = self.translator.resume()

            logger.info("response: {}", response)

            return quart.json.jsonify(response)
