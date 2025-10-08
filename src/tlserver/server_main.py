import inspect
import json
import logging
import signal
import uuid
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from time import monotonic

import quart
import trio
from hypercorn.config import Config
from hypercorn.trio import serve
from loguru import logger
from quart import Response, request
from quart_cors import cors
from quart_trio import QuartTrio

from tlserver.config import AppSettings
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


def translator_handler(translator: Translator) -> Callable[[], Awaitable[Response]]:
    async def receive_command() -> Response:
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

            response = None

            match message:
                case "close server":
                    die.set()

                case "check if server is ready":
                    response = translator.is_ready

                case "translate sentences":
                    if isinstance(content, str):
                        with timed("translated") as _:
                            translation = await translator.translate(content)
                        response = translation

                case "translate batch":
                    if isinstance(content, list):
                        with timed("batch translated") as _:
                            translation = await translator.translate_batch(content)
                        response = translation

                case "change input language":
                    if isinstance(content, str):
                        response = translator.change_input_language(content)

                case "change output language":
                    if isinstance(content, str):
                        response = translator.change_output_language(content)

                case "pause":
                    response = translator.pause()

                case "resume":
                    response = translator.resume()

            logger.info("response: {}", response)

            return quart.json.jsonify(response)

    return receive_command


# ===========================================================
# MAIN APPLICATION
# ===========================================================


TRANSLATOR_CLASSES: dict[str, type[Translator] | None] = {
    "Offline": OfflineTranslator,
    "Google": None,
    "LLM": LLMTranslator,
    "DeepL": None,
}


config = AppSettings()
logger.info(f"{config = }")

app = QuartTrio(__name__)
app = cors(app, allow_origin="*")
die = trio.Event()

# man why the hell does sugoi do per-port translators
TranslatorHandler = Callable[[], Awaitable[Response]]
handlers: dict[int, TranslatorHandler] = {}
for translator_config in config.translators:
    translator_cls = TRANSLATOR_CLASSES[translator_config.kind]
    if translator_cls:
        handlers[translator_config.port] = translator_handler(
            translator_cls(translator_config)
        )


@app.route("/", methods=["POST", "GET"])
async def dispatch() -> Response:
    # ASGI server info (host, port)
    host, port = request.scope.get("server", ("", 0))
    handler = handlers.get(port)
    if handler is None:
        return Response(
            f"No plugin for port {port}\n", status=404, mimetype="text/plain"
        )
    return await handler()


@app.before_serving
async def on_start() -> None:
    logger.info("Hello, starting up.")


@app.after_serving
async def on_stop() -> None:
    logger.info("Goodbye, shutting down.")


async def serve_handler(port: int) -> None:
    config = Config.from_mapping(
        bind=[f"0.0.0.0:{port}"],
        errorlog=None,
    )

    logger.debug("hypercorn serving")
    try:
        await serve(
            app,
            config,
            shutdown_trigger=die.wait,
        )
    finally:
        logger.debug("hypercorn stopping")


async def main() -> None:
    trio_token = trio.lowlevel.current_trio_token()

    def _handle_shutdown(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}; beginning graceful shutdown.")
        trio_token.run_sync_soon(die.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        if getattr(signal, sig.name, None) is not None:
            try:
                signal.signal(sig, _handle_shutdown)
            except ValueError as e:
                # SIGTERM is missing on Windows; ignore.
                logger.warning("Ignored error: {}", e)
                continue

    async with trio.open_nursery() as nursery:
        for port in handlers:
            nursery.start_soon(serve_handler, port)


if __name__ == "__main__":
    trio.run(main)
