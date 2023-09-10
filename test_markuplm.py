from markuplm_qa import MarkupLM, QADataset
import pandas as pd
import getopt
import sys

def main():
    #Command to run the code
    """
    python3 test.py --device="cuda" --test_ds="dataset.json" --output_csv="test_set_results.csv"
    """

    args = sys.argv[1:]

    oplist, args = getopt.getopt(args, "", ["test_ds=","device=","output_csv="])

    for op in oplist:
        if op[0] == "--test_ds":
            test_ds_path = op[1]
        elif op[0] == "--device":
            device = op[1]
        elif op[0] == "--output_csv":
            output_csv_path = op[1]
    
    markup_lm = MarkupLM("microsoft/markuplm-base-finetuned-websrc",device)
    dataset_dict = markup_lm.load_dataset(test_ds_path)
    
    #Code for evaluating the model on dataset
    
    answers_list = markup_lm.test(dataset_dict)
    df = pd.DataFrame.from_records(answers_list)
    df.to_csv(output_csv_path)


if __name__ == "__main__":
    main()