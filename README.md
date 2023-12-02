# Microsoft-MarkupLM

Description of model can be found at: https://medium.com/@hussainsyedshaharyaar/extractive-question-answering-using-bert-based-model-markuplm-798456e730ba

For running the inference use the following command:

`python3 test_markuplm.py --device="cuda" --test_ds="dataset.json" --output_csv="test_set_results.csv"`

Code can be run on cpu by setting "device" option to "cpu". Make sure to pass our own JSON-formatted dataset filepath in the command. Output answers for each question present in dataset are returned in generated CSV file.

JSON Structure of Dataset is as follows:

```
{
    "test_ds": [
                {"question" : "<your question goes here>", "html" : "<html code in which you have to search answer goes here>" , "answer" : "<ground truth answer goes here>"},
                {"question" : "<your question goes here>", "html" : "<html code in which you have to search answer goes here>" , "answer" : "<ground truth answer goes here>"},
                ...
                {"question" : "<your question goes here>", "html" : "<html code in which you have to search answer goes here>" , "answer" : "<ground truth answer goes here>"}
            ]
}
```

Original MarkupLM codebase can be found at: https://github.com/microsoft/unilm. In this repo, I have developed the code for running the inference over a dataset formatted in JSON with (question, document) and (answer) pairs. Moreover, I have developed my own custom dataset of ~50 samples for testing the model and evaluating how good it is.

MarkupLM Reference:

```
@misc{li2022markuplm,
      title={MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding}, 
      author={Junlong Li and Yiheng Xu and Lei Cui and Furu Wei},
      year={2022},
      eprint={2110.08518},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
