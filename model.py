import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

PRETRAINED_MODEL_MAP = {
    'bert':BertModel
}

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=config) # Load pretrained bert

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.e1_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(config.hidden_size * 3, config.num_labels, args.dropout_rate, use_activation=False)

    @staticmethod
    def entity_averate(hidden_ooutput, e_mask):
        """
        Average the entity hidden state vectors
        :param hidden_ooutput:  [batch_size, max_seq_len, dim]
        :param e_mask:  [batch_size, max_seq_len]
                e.g. e_mask[0] = [0,0,1,1,1,0,...,0]
        :return:  [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1) # [batch_size, 1, max_seq_len]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1) # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_ooutput).squeeze(1) # [batch_size, 1, dim]-->[batch_size, dim]
        avg_vector = sum_vector.float() / length_tensor.float()

        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # sequence_output, pooled_output,(hidden_states), (attention)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        #Average
        e1_h = self.entity_averate(sequence_output, e1_mask)
        e2_h = self.entity_averate(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        #Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs # loss, logits, hidden_states, attentions


