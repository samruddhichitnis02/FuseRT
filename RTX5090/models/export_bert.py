import torch
from transformers import BertModel


class BertWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained(
            "bert-base-uncased", attn_implementation="eager"
        ).eval()

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state, out.pooler_output


model = BertWrapper().eval()

ids = torch.randint(0, 30000, (1, 128))
mask = torch.ones(1, 128, dtype=torch.long)

torch.onnx.export(
    model,
    (ids, mask),
    "models/bert.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
        "pooler_output": {0: "batch"},
    },
    opset_version=16,
    dynamo=False,
)
print("saved models/bert.onnx")