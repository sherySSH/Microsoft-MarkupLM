import json
from transformers import AutoProcessor, MarkupLMForQuestionAnswering, MarkupLMTokenizerFast, MarkupLMFeatureExtractor, MarkupLMProcessor
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import logging

class QADataset(Dataset):
    def __init__(self, dataset_dict) -> None:
        super().__init__()
        self.dataset = dataset_dict

    def __getitem__(self, index):
        sample = self.dataset[index]
        question = sample["question"]
        html = sample["html"]
        annotated_answer = sample["answer"]

        return question, html, annotated_answer


class MarkupLM:
    def __init__(self, model_name, device) -> None:
        self.device = device
        self.processor , self.model = self._load_model(model_name)

    def load_dataset(self, filepath : 'str'):
        with open(filepath, 'r') as f:
            content = f.read()
            dataset_dict = json.loads(content)

        return dataset_dict


    def _pad(self, predicted_seq, annotated_seq):

        if len(predicted_seq) <= len(annotated_seq):

            pad_length = len(annotated_seq) - len(predicted_seq)
            #1 corresponds to <pad> token
            predicted_seq.extend( [1]*pad_length )
            return predicted_seq, annotated_seq

        elif len(predicted_seq) > len(annotated_seq):
            pad_length = len(predicted_seq) - len(annotated_seq)
            annotated_seq.extend( [1]*pad_length )

            return predicted_seq, annotated_seq


    def infer(self, question : 'str', html_source : 'str'):
        self.model = self.model.eval()

        encoding = self.processor(html_source, questions=question, return_tensors="pt").to(self.device)
        #we are only performing the forward-pass therefore we do not need to pre-allocate additional memory for gradients
        #hence we will set no_grad flag to true
        with torch.no_grad():
            outputs = self.model(**encoding)

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()

            predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
            answer = self.processor.decode(predict_answer_tokens).strip()

            return answer, predict_answer_tokens
        
    
    def forward(self, question : 'str', html_source : 'str'):
        self.model = self.model.train(True)

        encoding = self.processor(html_source, questions=question, return_tensors="pt").to(self.device)

        outputs = self.model(**encoding)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = self.processor.decode(predict_answer_tokens).strip()

        return answer, predict_answer_tokens


    def test(self, dataset_dict : 'dict'):

        test_set = dataset_dict["test_ds"]
        answer_list = []
        for sample in tqdm(test_set):
            question = sample["question"]
            html_source = sample["html"]
            annotated_answer = sample["answer"]
            answer, _ = self.infer(question, html_source)
            answer_list.append({ "question" : question , "html" : html_source ,"predicted_answer" : answer , "annotated_answer" : annotated_answer})

        return answer_list

    def train(self, dataset, epochs, optimizer, log_filepath):
        logging.basicConfig(filename=log_filepath, encoding='utf-8', level=logging.DEBUG)

        self.model = self.model.train(True)
        cross_ent_loss = nn.CrossEntropyLoss()
        counter = 0
        for epoch in tqdm(range(epochs)):
            for question, html, annotated_answer in dataset:
                
                annotated_answer_ids = self.processor(f"<html><body>{annotated_answer}</body></html>", add_special_tokens=False).input_ids[0]

                predicted_answer , prediction_input_ids = self.forward(question, html)
                prediction_input_ids = prediction_input_ids.cpu().detach().tolist()

                predicted_seq , annotated_seq = self._pad(prediction_input_ids, annotated_answer_ids)

                #one hot encoded version of annotated answer token ids
                annotated_seq_ohe = F.one_hot(torch.tensor( annotated_seq , dtype=torch.long) , 
                                                num_classes = self.model.config.vocab_size ) 

                #setting gradients flag to true because we need to calculate loss and perform backpropagation from loss function
                #all the way  back to weights of the first layer
                annotated_seq_ohe = annotated_seq_ohe.to(self.device).to(torch.float32)
                annotated_seq_ohe.requires_grad = True
                predicted_seq = torch.tensor(predicted_seq, dtype=torch.long).to(self.device)

                loss = cross_ent_loss(annotated_seq_ohe.to(torch.float32), torch.tensor(predicted_seq, dtype=torch.long) )
                logging.info(f"Iteration: {counter} ")
                logging.info(f"Predicted Seq. :  {predicted_seq} ")
                logging.info(f"Annotated Seq. :  {annotated_seq}" )
                logging.info(f"Predicted Ans. :  {predicted_answer} ")
                logging.info(f"Annotated Ans. :  {annotated_answer} ")
                logging.info(f"Loss : {round(loss.item() , 5)} ")
                logging.info(f"-------------------------------")
                #perform backpropagation to get gradient
                loss.backward()
                #updating the parameters values
                optimizer.step()
                #resetting gradients to zero, otherwise torch will continue accumulating the gradients after each backward pass
                optimizer.zero_grad()

                counter += 1



    def _load_model(self, model_name : 'str'):
        processor = AutoProcessor.from_pretrained(model_name)
        model = MarkupLMForQuestionAnswering.from_pretrained(model_name).to(self.device)

        return processor, model
