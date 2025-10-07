import re


def filter_text(extracted_text: str) -> str:
    result = extracted_text

    result = result.replace("\n", "<br>")
    result = result.replace("{", "")  # remove { sign in all positions of the text
    result = result.replace("�", "")  # remove the first unknown sign in the text
    result = re.sub(r"カ$", "", result)  # remove カ symbol at the end of the text
    result = re.sub(
        r"987$", "?", result
    )  # replace number 987 at the end of the text with ?
    result = re.sub(r"^:", "", result)  # remove colon at the beginning of the text

    return result  # noqa: RET504


def process_input_text(input_text: str) -> str:
    return filter_text(input_text)


def process_output_text(output_text: str) -> str:
    return filter_text(output_text)
