import os 
import argparse
import self_explain
from self_explain.preprocessing.store_parse_trees import ParsedDataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--splits", default=["train", "dev", "test"], type=str, nargs="+",
                        help="Which splits to process")
    parser.add_argument("--tokenizer_name", default='roberta-base', type=str, required=True,
                        help="Tokenizer name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite even if present")
    args = parser.parse_args()

    parsed_data = ParsedDataset(tokenizer_name=args.tokenizer_name, progress_bar=True)

    # Read input files from folder
    for file_split in args.splits:
        input_file_name = os.path.join(args.data_dir, file_split + '.tsv')
        if not os.path.exists(input_file_name):
            print(f"input file '{input_file_name}' not present, skipping")
            continue
        output_file_name = os.path.join(args.data_dir, file_split + '_with_parse.json')
        if not os.path.exists(output_file_name) or args.overwrite:
            print(f"storing parsed trees {input_file_name} -> {output_file_name}")
            parsed_data.read_and_store_from_tsv(input_file_name=input_file_name,
                                                output_file_name=output_file_name)
        else:
            print(f"output file '{output_file_name}' present, skipping. Use '--overwrite' option.")


if __name__ == "__main__": 
    main()
