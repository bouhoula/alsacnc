from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertPreTrainedModel,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput


def get_model(
    model_name: str,
    num_labels: int,
    model_type: str = "Bert",
    problem_type: str = "single_label_classification",
) -> BertPreTrainedModel:
    model_cls, config_cls = (
        (DistilBertForSequenceClassification, DistilBertConfig)
        if model_type == "DistilBert"
        else (BertForSequenceClassification, BertConfig)
    )
    config = config_cls.from_pretrained(
        model_name, problem_type=problem_type, num_labels=num_labels
    )
    return model_cls.from_pretrained(model_name, config=config)


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
