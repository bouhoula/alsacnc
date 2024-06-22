from typing import Optional

import click
import service_streamer.service_streamer
from flask import Flask, Response, jsonify, request
from service_streamer import ThreadedStreamer

from classifiers.text_classifiers.model_wrappers import IECTModelWrapper
from shared_utils import load_yaml

app = Flask(__name__)
model = None
streamer_ietc: Optional[ThreadedStreamer] = None


@app.route("/predict_ietc", methods=["POST"])
def stream_predict_ietc() -> Response:
    assert streamer_ietc is not None
    inputs = request.form.getlist("s")
    outputs = streamer_ietc.predict(inputs)
    return jsonify(outputs)


@click.command()
@click.option("--config_path", type=str)
def main(config_path: str) -> None:
    config = load_yaml(config_path)
    model_ietc = IECTModelWrapper(
        two_step=config["two_step_model"],
        problem_type=config["problem_type"],
        model_type=config["model_type"],
        model_path=config["model_path"],
        model_path_2=config["model_path_2"],
        tokenizer_path=config["tokenizer_path"],
        tokenizer_args=config["tokenizer_args"],
        label2id=config["label2id"],
    )

    global streamer_ietc
    streamer_ietc = ThreadedStreamer(model_ietc.predict, batch_size=8, max_latency=0.1)

    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    service_streamer.service_streamer.WORKER_TIMEOUT = 900
    main()
