import json

import litellm
from loguru import logger
from trio_asyncio import aio_as_trio

import plugins
from translator import Translator

# Load user settings
with open("User-Settings.json", encoding="utf-8") as settings_file:
    settings_data = json.load(settings_file)

    translator_settings = settings_data["Translation_API_Server"]["LLM"]


class LLMTranslator(Translator):
    def __init__(self):
        self.translator_ready_or_not = False
        self.can_change_language_or_not = True
        self.supported_languages_list = translator_settings["supported_languages_list"]
        self.input_language = self.supported_languages_list[
            translator_settings["input_language"]
        ]
        self.output_language = self.supported_languages_list[
            translator_settings["output_language"]
        ]
        self.model_name = translator_settings["model_name"]
        self.is_local = translator_settings["is_local"]
        self.api_key = translator_settings["api_key"]
        self.api_server = translator_settings["api_server"]
        self.context_lines = translator_settings["context_lines"]
        self.temperature = translator_settings["temperature"]
        self.top_p = translator_settings["top_p"]
        self.messages = []
        self.translator = ""
        self.stop_translation = False

        self._process_system_prompt()

    def _process_system_prompt(self):
        self.messages = []
        substitutions = {}
        if "{input_language}" in translator_settings["system_prompt"]:
            substitutions["input_language"] = self.input_language
        if "{output_language}" in translator_settings["system_prompt"]:
            substitutions["output_language"] = self.output_language
        # Apply formatting safely
        self.system = translator_settings["system_prompt"].format(**substitutions)
        self.messages.append({"role": "system", "content": self.system})

    @property
    def is_ready(self) -> bool:
        return self.translator_ready_or_not

    def pause(self) -> None:
        self.stop_translation = True

    def resume(self) -> None:
        self.stop_translation = False

    def activate(self) -> bool:
        self.translator_ready_or_not = True
        return self.translator_ready_or_not

    async def execute(self) -> str:
        response = ""
        if self.is_local:
            # response = await aio_as_trio(litellm.acompletion)(
            response = litellm.completion(
                model=self.model_name,
                messages=self.messages,
                api_key=self.api_key,
                api_base=self.api_server,
                temperature=self.temperature,
            )
        else:
            response = litellm.completion(
                model=self.model_name,
                messages=self.messages,
                api_key=self.api_key,
            )

        logger.debug("messages: {}", self.messages)

        return response.choices[0].message.content

    async def _translate(self, message: str) -> str:
        message = plugins.process_input_text(message)
        if self.stop_translation:
            return "Translation is paused at the moment"
        self.messages.append({"role": "user", "content": message})
        result = await self.execute()
        self.messages.append({"role": "assistant", "content": result})
        # Ensure only the last 10 user and assistant messages are kept
        # 10 user/assistant messages + 1 system message
        if len(self.messages) > self.context_lines:
            self.messages = [self.messages[0]] + self.messages[-10:]
        return plugins.process_output_text(result)

    async def translate(self, message: str) -> str:
        result = await self._translate(message)
        logger.info(f"{message!r}   ->   {result!r}")
        return result

    async def translate_batch(self, list_of_text_input: list[str]) -> list[str]:
        if self.stop_translation:
            return ["Translation is paused at the moment"]
        translation_list = []
        for text_input in list_of_text_input:
            translation = await self._translate(text_input)
            translation_list.append(translation)
        for original, translated in zip(
            list_of_text_input, translation_list, strict=True
        ):
            logger.info(f"{original!r}   ->   {translated!r}")
        return translation_list

    def check_if_language_available(self, language: str) -> bool:
        return self.supported_languages_list.get(language) is not None

    def change_output_language(self, output_language: str) -> str:
        if self.can_change_language_or_not:
            if self.check_if_language_available(output_language):
                self.output_language = output_language
                # Replace the system message in the messages list
                self._process_system_prompt()
                return f"output language changed to {output_language}"
            return "sorry, translator doesn't have this language"
        return "sorry, this translator can't change languages"

    def change_input_language(self, input_language: str) -> str:
        if self.can_change_language_or_not:
            if self.check_if_language_available(input_language):
                self.input_language = input_language
                # Replace the system message in the messages list
                self._process_system_prompt()
                return f"input language changed to {input_language}"
            return "sorry, translator doesn't have this language"
        return "sorry, this translator can't change languages"
