import json

import litellm

import plugins
from translator import Translator

# Load user settings
with open("User-Settings.json", encoding="utf-8") as file:
    user_settings = json.load(file)

translator_settings = user_settings["Translation_API_Server"]["LLM"]


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
        self.api_key = translator_settings["api_key"]
        self.api_server = translator_settings["api_server"]
        self.context_lines = translator_settings["context_lines"]
        self.temperature = translator_settings["temperature"]
        self.top_p = translator_settings["top_p"]
        self.messages = []
        self.translator = ""
        self.stop_translation = False

        # process system prompt
        substitutions = {}
        if "{input_language}" in translator_settings["system_prompt"]:
            substitutions["input_language"] = self.input_language
        if "{output_language}" in translator_settings["system_prompt"]:
            substitutions["output_language"] = self.output_language
        # Apply formatting safely
        self.system = translator_settings["system_prompt"].format(**substitutions)
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def pause(self) -> None:
        self.stop_translation = True

    def resume(self) -> None:
        self.stop_translation = False

    def activate(self) -> bool:
        self.translator_ready_or_not = True
        return self.translator_ready_or_not

    def execute(self) -> str | None:
        response = ""
        if any(
            name in self.model_name for name in ["ollama", "lm_studio", "oobabooga"]
        ):
            response = litellm.completion(
                model=self.model_name,
                messages=self.messages,
                api_key=self.api_key,
                api_base=self.api_server,
                temperature=self.temperature,
            )
            for message in self.messages:
                print(message)
        else:
            response = litellm.completion(
                model=self.model_name, messages=self.messages, api_key=self.api_key
            )

        # response = openai.chat.completions.create(
        #     model=self.model_name,
        #     messages=self.messages
        # )
        return response.choices[0].message.content

    def translate(self, message: str) -> str:
        message = plugins.process_input_text(message)
        if self.stop_translation:
            return "Translation is paused at the moment"
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        # Ensure only the last 10 user and assistant messages are kept
        if (
            len(self.messages) > self.context_lines
        ):  # 10 user/assistant messages + 1 system message
            self.messages = [self.messages[0]] + self.messages[-10:]
        result = plugins.process_output_text(result)
        return result

    def translate_batch(self, list_of_text_input: list[str]) -> list[str] | str:
        if self.stop_translation:
            return "Translation is paused at the moment"
        translation_list = []
        for text_input in list_of_text_input:
            translation = self.translate(text_input)
            translation_list.append(translation)
        return translation_list

    def check_if_language_available(self, language: str) -> bool:
        return self.supported_languages_list.get(language) is not None

    def change_output_language(self, output_language: str) -> str:
        if self.can_change_language_or_not:
            if self.check_if_language_available(output_language):
                self.output_language = output_language

                # Replace the system message in the messages list
                self.messages = []
                self.system = (
                    "You are a professional translator whose "
                    "primary goal is to precisely translate "
                    f"{self.input_language} to {self.output_language}. You can "
                    "speak colloquially if it makes the translation more "
                    f"accurate. Only respond in {self.output_language}. If you "
                    f"are unsure of a {self.input_language} sentence, still "
                    "always try your best estimate to respond with a complete "
                    f"{self.output_language} translation."
                )
                self.messages.append({"role": "system", "content": self.system})

                return f"output language changed to {output_language}"
            return "sorry, translator doesn't have this language"
        return "sorry, this translator can't change languages"

    def change_input_language(self, input_language: str) -> str:
        if self.can_change_language_or_not:
            if self.check_if_language_available(input_language):
                self.input_language = input_language

                # Replace the system message in the messages list
                self.messages = []
                self.system = (
                    "You are a professional translator whose primary goal is "
                    f"to precisely translate {self.input_language} to "
                    f"{self.output_language}. You can speak colloquially if it "
                    " akes the translation more accurate. Only respond in "
                    f"{self.output_language}. If you are unsure of a "
                    f"{self.input_language} sentence, still always try your "
                    "best estimate to respond with a complete "
                    f"{self.output_language} translation."
                )
                self.messages.append({"role": "system", "content": self.system})

                return f"input language changed to {input_language}"
            return "sorry, translator doesn't have this language"
        return "sorry, this translator can't change languages"


# Initialize the bot
# chat_gpt = Main_Translator()

# # Usage example
# if __name__ == "__main__":
#     # First request
#     response = chat_gpt.translate("──空気が動いた。")
#     print("Response 1:", response)

#     chat_gpt.change_output_language("Vietnamese")

#     # Second request
#     response = chat_gpt.translate("それまでボクを包み込んでいた青白い光が遠のいていく。")
#     print("Response 2:", response)

#     # Third request
#     response = chat_gpt.translate("ボクの右腕は、手首から先が剥き出しの機械部品でできている。")
#     print("Response 3:", response)
