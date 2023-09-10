from transformers import MarkupLMFeatureExtractor, MarkupLMTokenizerFast, MarkupLMProcessor
from transformers import MarkupLMModel
import torch.nn as nn
import torch

feature_extractor = MarkupLMFeatureExtractor()
tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base")
processor = MarkupLMProcessor(feature_extractor, tokenizer)

html_string = """
 <!DOCTYPE html>
 <html>
 <head>
 <title>Hello world</title>
 </head>
 <body>
 <h1 class="relative group">
	<a 
		id="welcome" 
		class="header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full" 
		href="#welcome"
	>
		<span><IconCopyLink/></span>
	</a>
	<span>
		Welcome
	</span>
</h1>

 <p>Here is my website.</p>
 </body>
 </html>"""


features = feature_extractor(html_string)
print(features)
# note that you can also add provide all tokenizer parameters here such as padding, truncation
encoding = processor(html_string, return_tensors="pt")

# print(encoding)
# # print(encoding)
# # print(encoding.keys())


model = MarkupLMModel.from_pretrained("microsoft/markuplm-base")
print(model)

outputs = model(**encoding)
last_hidden_states = outputs.last_hidden_state.squeeze(0)

repeat = last_hidden_states.shape[0]

x = last_hidden_states.repeat((repeat,1))
y = torch.repeat_interleave(last_hidden_states, repeat, dim=0)
cos = nn.CosineSimilarity(dim=1)
sim = cos(x,y)
print(torch.reshape(sim,(repeat,-1)))
# list(last_hidden_states.shape)

# logits = last_hidden_states.cpu().detach()
# print(last_hidden_states.shape)

# print(logits.argmax(-1).shape)