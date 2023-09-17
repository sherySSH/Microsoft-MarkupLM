# Microsoft-MarkupLM

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
