import os 
import argparse
import self_explain
from self_explain.preprocessing.store_parse_trees import ParsedDataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--splits", default=["train", "dev"], type=str, nargs="+",
                        help="Which splits to process")
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str, required=True,
                        help="Tokenizer name")

    args = parser.parse_args()

    parsed_data = ParsedDataset(tokenizer_name=args.tokenizer_name, progress_bar=True)

    # Read input files from folder
    for file_split in args.splits:
        input_file_name = os.path.join(args.data_dir, file_split + '.tsv')
        output_file_name = os.path.join(args.data_dir, file_split + '_with_parse.json')
        print(f"storing parsed trees {input_file_name} -> {output_file_name}")
        parsed_data.read_and_store_from_tsv(input_file_name=input_file_name,
                                            output_file_name=output_file_name)


if __name__ == "__main__": 
    main()
