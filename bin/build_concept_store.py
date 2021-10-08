import os 
import argparse
import logging
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
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")

    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    concept_store(input_file_name=args.input_train_file,
                  output_folder=args.output_folder,
                  model_name=args.model_name,
                  max_concept_length=args.max_concept_len)


if __name__ == "__main__":
    main()
