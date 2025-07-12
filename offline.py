import json

import ctranslate2
import sentencepiece as spm

import plugins
from translator import Translator

# ===========================================================
# INITIALIATION
# ===========================================================
user_settings_file = open("User-Settings.json", encoding="utf-8")
user_settings_data = json.load(user_settings_file)

offline_settings_data = user_settings_data["Translation_API_Server"]["Offline"]

port = offline_settings_data["HTTP_port_number"]
gpu = offline_settings_data["gpu"]
device = offline_settings_data["device"]  # cuda or cpu
intra_threads = offline_settings_data["intra_threads"]
inter_threads = offline_settings_data["inter_threads"]
beam_size = offline_settings_data["beam_size"]
repetition_penalty = offline_settings_data["repetition_penalty"]
silent = offline_settings_data["silent"]
disable_unk = offline_settings_data["disable_unk"]

model_dir = "./Sugoi_Model/ct2Model/"
sp_source_model = "./Sugoi_Model/spmModels/spm.ja.nopretok.model"
sp_target_model = "./Sugoi_Model/spmModels/spm.en.nopretok.model"

# ===========================================================
# MAIN APPLICATION
# ===========================================================

# translator = ctranslate2.Translator(modelDir, device=device, intra_threads=intra_threads, inter_threads=inter_threads)


def tokenize_batch(text: list[str] | str, sp_source_model: str) -> list[list[str]]:
    sp = spm.SentencePieceProcessor(sp_source_model)  # type: ignore
    if isinstance(text, list):
        return sp.encode(text, out_type=str)  # type: ignore
    return [sp.encode(text, out_type=str)]  # type: ignore


def detokenize_batch(text: list[list[str]], sp_target_model: str) -> list[str]:
    sp = spm.SentencePieceProcessor(sp_target_model)  # type: ignore
    return sp.decode(text)  # type: ignore


class OfflineTranslator(Translator):
    def __init__(self) -> None:
        self.translator_ready_or_not = False
        self.can_change_language_or_not = False
        self.supported_languages_list = {"English": "English", "Japanese": "Japanese"}
        self.input_language = self.supported_languages_list["Japanese"]
        self.output_language = self.supported_languages_list["English"]
        self.translator: ctranslate2.Translator | None = None
        self.stop_translation = False

    def pause(self) -> None:
        self.stop_translation = True

    def resume(self) -> None:
        self.stop_translation = False

    def activate(self) -> bool:
        self.translator = ctranslate2.Translator(
            model_dir,
            device=device,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
        )
        self.translator_ready_or_not = True
        return self.translator_ready_or_not

    def translate(self, message: str) -> str:
        if self.stop_translation:
            return "Translation is paused at the moment"

        if isinstance(message, list):
            return self.translate_batch(message)

        message = plugins.process_input_text(message)

        translated = self.translator.translate_batch(
            source=tokenize_batch(message, sp_source_model),
            beam_size=beam_size,
            num_hypotheses=1,
            return_alternatives=False,
            disable_unk=disable_unk,
            replace_unknowns=False,
            no_repeat_ngram_size=repetition_penalty,
        )

        final_result = []
        for result in translated:
            detokenized = "".join(
                detokenize_batch(result.hypotheses[0], sp_target_model)
            )
            final_result.append(detokenized)

        if isinstance(message, list):
            return final_result
        result = plugins.process_output_text(final_result[0])
        return result

    def translate_batch(self, list_of_text_input: list[str]) -> list[str] | str:
        if self.stop_translation:
            return "Translation is paused at the moment"
        translated = self.translator.translate_batch(
            source=tokenize_batch(list_of_text_input, sp_source_model),
            beam_size=beam_size,
            num_hypotheses=1,
            return_alternatives=False,
            disable_unk=disable_unk,
            replace_unknowns=False,
            no_repeat_ngram_size=repetition_penalty,
        )

        final_result = []
        for result in translated:
            detokenized = "".join(
                detokenize_batch(result.hypotheses[0], sp_target_model)
            )
            final_result.append(detokenized)

        if isinstance(list_of_text_input, list):
            return final_result
        return final_result[0]

    def check_if_language_available(self, language: str) -> bool:
        return self.supported_languages_list.get(language) is not None

    def change_output_language(self, output_language: str) -> str:
        if self.can_change_language_or_not:
            if self.check_if_language_available(output_language):
                self.output_language = output_language
                return f"output language changed to {output_language}"
            return "sorry, translator doesn't have this language"
        return "sorry, this translator can't change languages"

    def change_input_language(self, input_language: str) -> str:
        if self.can_change_language_or_not:
            if self.check_if_language_available(input_language):
                self.input_language = input_language
                return f"input language changed to {input_language}"
            return "sorry, translator doesn't have this language"
        return "sorry, this translator can't change languages"


# sugoi_translator = Sugoi_Translator()
# sugoi_translator.activate()
# print(sugoi_translator.change_input_language("Vietnamese"))
# print(sugoi_translator.translate("たまに閉じているものがあっても、中には何も入っていなかった。"))
