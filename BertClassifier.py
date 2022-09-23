
import torch
from torch import nn 
from transformers import BertModel, BertForSequenceClassification

class BertClassifier(nn.Module):

    def __init__(self, pretrained, hidden_size, num_labels): # , dropout=0.5):
        super(BertClassifier, self).__init__()
    
        self.num_labels = num_labels
        self.model = BertModel.from_pretrained(pretrained, output_hidden_states=True)
        # self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        document = self.model(
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

        document_emb = document[1]    
        logits = self.classifier(document_emb)
               
        loss = None
        if labels is not None:
            ce_loss_fct = nn.CrossEntropyLoss()
            loss = ce_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits