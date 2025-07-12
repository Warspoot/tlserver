import json
import time

from flask import Flask, request
from flask_cors import CORS, cross_origin
from waitress import serve

from llm import LLMTranslator
from offline import OfflineTranslator
from translator import Translator

# ===========================================================
# INITIALIATION
# ===========================================================
user_settings_file = open("../../../../User-Settings.json", encoding="utf-8")
user_settings_data = json.load(user_settings_file)

current_translator = user_settings_data["Translation_API_Server"]["current_translator"]
port = user_settings_data["Translation_API_Server"][current_translator][
    "HTTP_port_number"
]
host = "0.0.0.0"


# ===========================================================
# MAIN APPLICATION
# ===========================================================

translator: Translator = OfflineTranslator()
translator.activate()
# translator = Translator_API(Sugoi_Translator())


app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/", methods=["POST", "GET"])
@cross_origin()
def send_sugoi() -> None | str:
    tic = time.perf_counter()
    data = request.get_json(True)
    message = data.get("message")
    content = data.get("content")

    match message:
        case "close server":
            if current_translator == "DeepL":
                translator.close()
            shutdown_server()
            return None

        case "check if server is ready":
            result = translator.translator_ready_or_not
            return json.dumps(result)

        case "translate sentences":
            start = time.time()
            print("translation request received")
            translation = translator.translate(content)
            print(translation)
            end = time.time()
            print(end - start)
            return json.dumps(translation, ensure_ascii=False)

        case "translate batch":
            print("translation request received")
            translation = translator.translate_batch(content)
            return json.dumps(translation, ensure_ascii=False)

        case "change input language":
            return json.dumps(translator.change_input_language(content))

        case "change output language":
            return json.dumps(translator.change_output_language(content))

        case "pause":
            return json.dumps(translator.pause())

        case "resume":
            return json.dumps(translator.resume())


def shutdown_server() -> None:
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


# if __name__ == "__main__":
#     #app.run(host=host, port=port)
#     serve(app, host=host, port=port)

serve(app, host=host, port=port)
