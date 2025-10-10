import inspect
import logging
import signal
from contextlib import suppress
from io import StringIO

import trio
from hypercorn.config import Config
from hypercorn.trio import serve
from loguru import logger
from quart import Response, request
from quart_cors import cors
from quart_trio import QuartTrio
from rich.console import Console
from rich.pretty import Pretty

from tlserver.config import AppSettings
from tlserver.handler import LegacyTranslatorHandler
from tlserver.translator import Translator
from tlserver.translators.llm import LLMTranslator
from tlserver.translators.offline import OfflineTranslator

# ===========================================================
# INITIALIATION
# ===========================================================


def rich_str(obj: object) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor")
    console.print(Pretty(obj))
    return buf.getvalue().rstrip()


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
logger.info(f"Config loaded:\n{rich_str(config.model_dump())}")

app = QuartTrio(__name__)
app = cors(app, allow_origin="*")
die = trio.Event()

# man why the hell does sugoi do per-port translators
handlers: dict[int, LegacyTranslatorHandler] = {}
for translator_config in config.translators:
    translator_cls = TRANSLATOR_CLASSES[translator_config.kind]
    if translator_cls:
        with suppress(Exception):
            handlers[translator_config.port] = LegacyTranslatorHandler(
                translator_cls(translator_config)
            )


# TODO: custom router instead?
@app.route("/", methods=["POST", "GET"])
async def dispatch() -> Response:
    # ASGI server info (host, port)
    # wow i love python is there no better way to do this
    port = handler = None
    server: tuple[str, int | None] | None = request.scope.get("server", None)
    if server is not None:
        port: int | None = server[1]
    if port is not None:
        handler = handlers.get(port)
    if handler is None:
        return Response(
            f"No plugin for port {port}\n", status=404, mimetype="text/plain"
        )
    return await handler.receive_command()


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


async def amain() -> None:
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


def main() -> None:
    trio.run(amain)
