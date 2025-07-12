from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def pause(self) -> None:
        pass

    @abstractmethod
    def resume(self) -> None:
        pass

    @abstractmethod
    def activate(self) -> bool:
        pass

    # @abstractmethod
    # def execute(self) -> str | None:
    #     pass

    @abstractmethod
    def translate(self, message: str) -> str:
        pass

    @abstractmethod
    def translate_batch(self, list_of_text_input: list[str]) -> list[str] | str:
        pass

    @abstractmethod
    def check_if_language_available(self, language: str) -> bool:
        pass

    @abstractmethod
    def change_output_language(self, output_language: str) -> str:
        pass

    @abstractmethod
    def change_input_language(self, input_language: str) -> str:
        pass
