# TODO: refactor this so that the user can configure this in config
# for now this does nothing


def filter_text(extracted_text: str) -> str:
    result = extracted_text

    return result  # noqa: RET504


def process_input_text(input_text: str) -> str:
    return filter_text(input_text)


def process_output_text(output_text: str) -> str:
    return filter_text(output_text)
