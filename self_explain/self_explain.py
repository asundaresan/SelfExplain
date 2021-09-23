import torch
import logging 
import os
import datetime
import numpy as np
import csv
import re

from .model.SE_XLNet import SEXLNet
from .model.infer_model import gil_interpret, lil_interpret, load_concept_map, load_dev_examples
from .model.data import ClassificationData
from .preprocessing.store_parse_trees import ParsedDataset

def convert_to_sentences(text, convert=False, label=0) -> list:
    """ Process text into sentences
    Args: 
        text (str)

    Returns:
        list of dict each with 'sentence' and 'label' key
    """
    if convert:
        text = re.sub(r"(\.)", r"\1\n", text)
        text = re.sub(r"(\?)", r"\1\n", text)
        text = re.sub(r"(\!)", r"\1\n", text)
        text = re.sub(r"(\.|\?|\!|')", r" \1", text)
        sentences = [s for s in text.split("\n") if len(s.split()) > 0]
    else:
        sentences = [text,]
    data = [dict(sentence=sentence, label=label) for sentence in sentences]
    return data 



class SelfExplainCharacterizer(object):
    def __init__(self, checkpoint=None, concept_map_filename=None, **kwargs):
        parser_tokenizer_name = kwargs.get("parser_tokenizer", "xlnet-base-cased")
        if checkpoint is None:
            raise RuntimeError(f"ImageCaptionCharacterizer model file missing!")
        print(f"- loading checkpoint from {checkpoint}")
        self.model = SEXLNet.load_from_checkpoint(checkpoint)
        self.model.eval()
        print(f"- loading concept map from {concept_map_filename}")
        self.concept_map = load_concept_map(concept_map_filename)
        print(f"- parser tokenizer: {parser_tokenizer_name}")
        self.parsed_data = ParsedDataset(tokenizer_name=parser_tokenizer_name)
        self.save_dir = os.path.join("output", datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S"))
        print(f"- save_dir: {self.save_dir}")
        self.count = 0


    def compute_parse_tree(self, data: list) -> str:
        """ Store in tsv format and parse into trees.
        Return:
            str: JSON filename where the parse tree is stored
        """
        data_dir = os.path.join(self.save_dir, f"{self.count:06d}")
        self.count += 1
        if not os.path.exists(data_dir):
            print(f"creating working dir: {data_dir}")
            os.makedirs(data_dir)
        input_filename = os.path.join(data_dir, "dev.tsv")
        output_filename = os.path.join(data_dir, "dev_with_parse.json")
        # write tsv
        print(f"  writing text to {input_filename}")
        with open(input_filename, "w") as handle:
            writer = csv.DictWriter(handle, ["sentence", "label"], delimiter="\t")
            writer.writeheader()
            writer.writerows(data)
        print(f"  writing parsed tree to {output_filename}")
        self.parsed_data.read_and_store_from_tsv(input_file_name=input_filename, output_file_name=output_filename)
        return output_filename


    def process(self, text: str, convert=False):
        """ Process image and caption together to get an LLR value.

        Args: 
            text (str): text to process 
            convert (bool): Flag to indicate if text should be converted into sentences.
                    If using SST-2, it is not necessary, otherwise the text should be split into sentences
                    and processsed appropriately.
        """
        data = convert_to_sentences(text, convert=convert)
        # get location of parse_tree_filename
        parse_tree_filename = self.compute_parse_tree(data)
        batch_size = min(256, len(data))
        results = self.evaluate(parse_tree_filename, batch_size=batch_size)
        return results


    def evaluate(self, parse_tree_filename, batch_size=1):
        # load dev samples from file 
        samples = load_dev_examples(parse_tree_filename)
        # split it into batches to match dataloader
        dev_samples_batches = [samples[start:start+batch_size] for start in range(0, len(samples), batch_size)]
        # get dataloader 
        basedir = os.path.dirname(parse_tree_filename)
        dm = ClassificationData(basedir=basedir, tokenizer_name=self.model.hparams.model_name, batch_size=batch_size)
        dataloader = dm.val_dataloader()
        # initialize result 
        result = dict(samples=samples, predicted_labels=[], gil_interpretations=[], lil_interpretations=[])
        with torch.no_grad():
            for batch, dev_samples in zip(dataloader, dev_samples_batches):
                input_tokens, token_type_ids, nt_idx_matrix, labels = batch
                logits, acc, interpret_dict_list = self.model(batch)
                gil_interpretations = gil_interpret(concept_map=self.concept_map,
                                                    list_of_interpret_dict=interpret_dict_list)
                # note that dev_samples match the logits so current_idx is set to 0
                lil_interpretations = lil_interpret(logits=logits,
                                                    list_of_interpret_dict=interpret_dict_list,
                                                    dev_samples=dev_samples,
                                                    current_idx=0)
                # note that acc is meaningless as the labels are meaningless
                #accs.append(acc)
                # Note that these labels are meaningless
                #true_labels.extend(labels.tolist())
                predicted_labels = torch.argmax(logits, -1).tolist()
                result["predicted_labels"].extend(predicted_labels)
                result["gil_interpretations"].extend(gil_interpretations)
                result["lil_interpretations"].extend(lil_interpretations)

