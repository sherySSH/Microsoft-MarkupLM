# Microsoft-MarkupLM

For running the inference use the following command:

`python3 test.py --device="cuda" --test_ds="dataset.json" --output_csv="test_set_results.csv"`

Code can be run on cpu by setting "device" option to "cpu". Make sure to pass our own JSON-formatted dataset filepath in the command. Output answers for each question present in dataset are returned in generated CSV file.
