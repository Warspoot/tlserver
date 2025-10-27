# TODO: refactor this so that the user can configure this in config
# for now this does nothing
import re
from loguru import logger


def filter_text(extracted_text: str) -> str:
    result = extracted_text

    return result  # noqa: RET504


def process_input_text(input_text: str, dictionary: dict[str, str] | None = None) -> str:

    logger.debug(f"Original text: {input_text!r}") #logging original text

    if dictionary:
        for jp, en in dictionary.items():
            input_text = input_text.replace (jp, en)

    logger.debug(f"Processed text: {input_text!r}") #logging processed text
    return filter_text(input_text)


def process_output_text(output_text: str) -> str:
    return filter_text(output_text)
