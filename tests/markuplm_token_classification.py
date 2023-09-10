from transformers import AutoProcessor, AutoModelForTokenClassification
import torch

device="cuda"

processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
processor.parse_html = False
model = AutoModelForTokenClassification.from_pretrained("microsoft/markuplm-base", num_labels=7).eval().to(device)

nodes = ["hello", "world"]
xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"]
node_labels = [1, 2]
encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")
encoding = encoding.to(device)

# print(encoding)

with torch.no_grad():
    outputs = model(**encoding)

loss = outputs.loss
logits = outputs.logits

print(model.config.id2label)