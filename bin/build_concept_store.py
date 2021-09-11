import os 
import argparse
import self_explain 
from self_explain.preprocessing.build_concept_store import concept_store 

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_train_file", "-i", default=None, type=str, required=True,
                        help="The input train file")

    parser.add_argument("--output_folder", "-o", default='roberta-base', type=str, required=True,
                        help="Output folder for concept store and dict")

    parser.add_argument("--model_name", "-m", default='roberta-base', type=str, required=True,
                        help="Model name")

    parser.add_argument("--max_concept_len", "-l", default=5, type=int, required=True,
                        help="Max length of concept")

    args = parser.parse_args()

    concept_store(input_file_name=args.input_train_file,
                  output_folder=args.output_folder,
                  model_name=args.model_name,
                  max_concept_length=args.max_concept_len)


if __name__ == "__main__":
    main()
