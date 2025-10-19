from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tlserver.config import TranslatorSettingsBase

ConfigT = TypeVar("ConfigT", bound=TranslatorSettingsBase)


class Translator(ABC, Generic[ConfigT]):
    config: ConfigT

    @abstractmethod
    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @property
    @abstractmethod
    def is_ready(self) -> bool:
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

    @abstractmethod
    async def translate(self, message: str) -> str:
        pass

    @abstractmethod
    async def translate_batch(self, list_of_text_input: list[str]) -> list[str]:
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
