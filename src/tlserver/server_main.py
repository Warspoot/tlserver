import inspect
import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from time import monotonic

import trio
import trio_asyncio
from hypercorn.config import Config
from hypercorn.trio import serve
from loguru import logger
from quart import request
from quart_cors import cors
from quart_trio import QuartTrio

from tlserver.translator import Translator
from tlserver.translators.llm import LLMTranslator
from tlserver.translators.offline import OfflineTranslator

# ===========================================================
# INITIALIATION
# ===========================================================


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


@contextmanager
def timed(action):
    tick = monotonic()
    try:
        yield
    finally:
        tock = monotonic()
        logger.info(f"{action} in {tock - tick:.3f}s")


@dataclass
class Settings:
    current_translator: str
    port: int
    host: str


settings: Settings
with open("User-Settings.json", encoding="utf-8") as settings_file:
    settings_data = json.load(settings_file)["Translation_API_Server"]
    current_translator = settings_data["current_translator"]
    settings = Settings(
        current_translator=current_translator,
        port=settings_data[current_translator]["HTTP_port_number"],
        host="0.0.0.0",
    )

logger.info(f"{settings = }")

# ===========================================================
# MAIN APPLICATION
# ===========================================================

translator: Translator = OfflineTranslator()
match settings.current_translator:
    case "LLM":
        translator = LLMTranslator()
    case "Offline":
        translator = OfflineTranslator()
    case _:
        pass
translator.activate()


app = QuartTrio(__name__)
app = cors(app, allow_origin="*")
die = trio.Event()


@app.route("/", methods=["POST", "GET"])
async def receive_command() -> str:
    with logger.contextualize(uid=uuid.uuid4()), timed("command handled") as _:
        data = await request.get_json(force=True)
        message: str = data.get("message")
        content = data.get("content")

        logger.info(f"received command '{message}'  |  {content}")

        # special case for legacy support
        if message == "translate sentences" and isinstance(content, list):
            logger.debug(
                "legacy support: translate single sentence converted to batch translate"
            )
            message = "translate batch"

        response = ""

        match message:
            case "close server":
                die.set()

            case "check if server is ready":
                response = json.dumps(translator.is_ready)

            case "translate sentences":
                if isinstance(content, str):
                    with timed("translated") as _:
                        translation = await translator.translate(content)
                    response = json.dumps(translation, ensure_ascii=False)

            case "translate batch":
                if isinstance(content, list):
                    with timed("batch translated") as _:
                        translation = await translator.translate_batch(content)
                    response = json.dumps(translation, ensure_ascii=False)

            case "change input language":
                if isinstance(content, str):
                    response = json.dumps(translator.change_input_language(content))

            case "change output language":
                if isinstance(content, str):
                    response = json.dumps(translator.change_output_language(content))

            case "pause":
                response = json.dumps(translator.pause())

            case "resume":
                response = json.dumps(translator.resume())

        logger.info("response: {}", response)

        return response


if __name__ == "__main__":
    trio_asyncio.run(
        partial(
            serve,
            app,
            Config.from_mapping(
                bind=[f"{settings.host}:{settings.port}"],
                errorlog=None,
            ),
            shutdown_trigger=die.wait,
        )
    )
