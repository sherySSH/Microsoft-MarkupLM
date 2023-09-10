from transformers import AutoProcessor, MarkupLMForQuestionAnswering
import torch

device = "cuda"

processor = AutoProcessor.from_pretrained("microsoft/markuplm-base-finetuned-websrc")
model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base-finetuned-websrc")

model = model.eval().to(device)

html_string = """
                <html> 
                    <head>
                        <title>Poll: most Americans say gun ownership increases safety. Research: no. - Vox</title>
                        <meta property="author" content="German Lopez">
                    </head>
                </html>
            """

question = "What is property?"

encoding = processor(html_string, questions=question, return_tensors="pt")
encoding = encoding.to(device)

print(encoding)

with torch.no_grad():
    outputs = model(**encoding)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
answer = processor.decode(predict_answer_tokens).strip()
print("Answer :",answer)


# print(processor.decode([0,  2264,    16,     5,   766,     9, 21032,     2,   597,  8831, 21032,   133, 16874,   102,    12,  7293, 21032, 25941,   132, 43925, 12,   250,  5479,   819, 29829,   878,    23,    62,     7,   132, 4,   245, 23108,    13,  4543,  3340,   906,  1553,  1263,     8, 819, 26487,   597,  8831, 22794,   133,    70,    12,  4651, 10617, 16193,    12,   534,  4671, 22794,  1575,    62,     7,   361,   207, 3845,  6548,   819,     4, 50118,  1437,  1437,  1437,  1437,  1437, 1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437, 1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437, 17586,    13, 1007,    12, 24645,     6,   239,    12, 15526,  6548, 26487,   245, 4377,   211,  4629,  3908, 29614, 10646, 32408,   806,     6,     5, 588,  1794,   361,  1698,  2744,    64, 42972,  3003,    63, 10646, 7,    62,     7,   508,  4377,     4, 50118,  1437,  1437,  1437, 1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437, 1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437,  1437, 407,    47,    64,  5405,   227,  3798,   818, 10062, 43320, 26487, 20645,  2650, 39421, 48097,    62,     7,  1718,   207,    55,  1254, 1118,     7,     5,   986, 11709, 40904, 10655, 26487,     2]))