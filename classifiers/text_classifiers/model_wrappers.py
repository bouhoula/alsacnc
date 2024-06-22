from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from classifiers.text_classifiers.metrics import get_prediction
from classifiers.text_classifiers.models import get_model


class IECTModelWrapper:
    def __init__(
        self,
        two_step: bool,
        problem_type: str,
        model_type: str,
        model_path: str,
        label2id: Dict[str, int],
        tokenizer_path: str,
        tokenizer_args: Dict,
        model_path_2: Optional[str] = None,
    ):
        self.two_step = two_step
        self.problem_type = problem_type
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer_args = tokenizer_args
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Running on", self.device)
        if two_step:
            self.detection_model = get_model(
                model_path, 2, model_type, "single_label_classification"
            )
            self.detection_model.to(self.device)
            assert model_path_2 is not None
            self.classification_model = get_model(
                model_path_2, len(label2id), model_type, problem_type
            )
            self.classification_model.to(self.device)
        else:
            self.model = get_model(model_path, len(label2id), model_type, problem_type)
            self.model.to(self.device)
        self.label2id = label2id
        self.id2label = {label2id[key]: key for key in label2id}

    def predict(self, batch: List[str]) -> List[str]:
        input_ids = []
        attentions_masks = []
        for text in batch:
            encoded_text = self.tokenizer.encode_plus(text, **self.tokenizer_args)
            input_ids.append(encoded_text["input_ids"])
            attentions_masks.append(encoded_text["attention_mask"])

        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attentions_masks, device=self.device)

        if self.two_step:
            detection_pred = get_prediction(
                self.detection_model(input_ids, attention_mask).logits.detach().cpu()
            )
            classification_pred = get_prediction(
                self.classification_model(input_ids, attention_mask)
                .logits.detach()
                .cpu()
            )
            return [
                "Other" if not d_pred else self.id2label[c_pred]
                for (d_pred, c_pred) in zip(
                    detection_pred.tolist(), classification_pred.tolist()
                )
            ]
        else:
            return [
                self.id2label[p]
                for p in get_prediction(
                    self.model(input_ids, attention_mask).logits.detach().cpu()
                ).tolist()
            ]
