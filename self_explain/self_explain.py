import torch
import logging 
import os
import datetime
import numpy as np
import csv
import re
import spacy

from typing import Tuple, List

from .model.SE_XLNet import SEXLNet
from .model.infer_model import gil_interpret, lil_interpret, load_concept_map, load_dev_examples
from .model.data import ClassificationData
from .preprocessing.store_parse_trees import ParsedDataset



class SelfExplainCharacterizer(object):
    def __init__(self, checkpoint_filename=None, concept_map_filename=None, **kwargs):
        self.nlp = spacy.load("en_core_web_sm")
        # get override parameters for load_from_checkpoint (concept_store, hparams, etc)
        checkpoint_kwargs = kwargs.get("checkpoint_kwargs", {})
        # what tokenizer to use 
        parser_tokenizer_name = kwargs.get("parser_tokenizer", "xlnet-base-cased")
        if checkpoint_filename is None:
            raise RuntimeError(f"checkpoint_filename=None, but it should be specified!")
        print(f"- loading checkpoint from {checkpoint_filename}")
        self.model = SEXLNet.load_from_checkpoint(checkpoint_filename, **checkpoint_kwargs)
        self.model.eval()
        if concept_map_filename is None:
            raise RuntimeError(f"concept_map_filename=None, but it should be specified!")
        print(f"- loading concept map from {concept_map_filename}")
        self.concept_map = load_concept_map(concept_map_filename)
        print(f"- parser tokenizer: {parser_tokenizer_name}")
        self.parsed_data = ParsedDataset(tokenizer_name=parser_tokenizer_name)
        self.save_dir = os.path.join("output", datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S"))
        print(f"- save_dir: {self.save_dir}")
        self.count = 0


    def to_sentences(self, text) -> List[str]:
        """ Convert input into sentences
        """
        doc = self.nlp(text)
        sentences = []
        for sentence in doc.sents:
            sentences.append(sentence.text)
        return sentences


    def compute_parse_tree(self, data: list) -> str:
        """ Store in tsv format and parse into trees.
        Return:
            str: JSON filename where the parse tree is stored
        """
        data_dir = os.path.join(self.save_dir, f"{self.count:06d}")
        self.count += 1
        if not os.path.exists(data_dir):
            logging.info(f"creating working dir: {data_dir}")
            os.makedirs(data_dir)
        input_filename = os.path.join(data_dir, "dev.tsv")
        output_filename = os.path.join(data_dir, "dev_with_parse.json")
        # write tsv
        logging.info(f"  writing text to {input_filename}")
        with open(input_filename, "w") as handle:
            writer = csv.DictWriter(handle, ["sentence", "label"], delimiter="\t")
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"  writing parsed tree to {output_filename}")
        self.parsed_data.read_and_store_from_tsv(input_file_name=input_filename, output_file_name=output_filename)
        return output_filename


    def process(self, text: str, batch_size=32, label=0):
        """ Process text to get probability and evidence

        Args: 
            text (str): text to process 
        """
        data = [dict(sentence=sentence, label=label) for sentence in self.to_sentences(text)]
        # get location of parse_tree_filename
        parse_tree_filename = self.compute_parse_tree(data)
        batch_size = min(batch_size, len(data))
        result = self.evaluate(parse_tree_filename, batch_size=batch_size)
        prob = max(result["scores"])
        evidence = dict()
        return prob, evidence


    def evaluate(self, parse_tree_filename, batch_size=1):
        # load dev samples from file 
        samples = load_dev_examples(parse_tree_filename)
        logging.debug(f"loaded {len(samples)} from {parse_tree_filename}")
        # split it into batches to match dataloader
        dev_samples_batches = [samples[start:start+batch_size] for start in range(0, len(samples), batch_size)]
        # get dataloader 
        basedir = os.path.dirname(parse_tree_filename)
        dm = ClassificationData(basedir=basedir, tokenizer_name=self.model.hparams.model_name, batch_size=batch_size, num_workers=0)
        dataloader = dm.val_dataloader()
        # initialize result 
        result = dict(samples=samples, predicted_labels=[], true_labels=[], scores=[], gil_interpretations=[], lil_interpretations=[])
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
                # note that acc, true_labels is meaningless as the labels are meaningless
                #accs.append(acc)
                # XXX TODO this should return a value between 0 and 1 so that we can apply a threshold
                pred_labels = torch.argmax(logits, -1).tolist()
                pred_scores = torch.softmax(logits, -1).tolist()
                # get the score for label 1
                scores = [ps[1] for ps, pl in zip(pred_scores, pred_labels)]
                logging.info(f"scores={scores}, pred_labels={pred_labels}, pred_scores={pred_scores}")
                result["predicted_labels"].extend(pred_labels)
                result["true_labels"].extend(labels.tolist())
                result["scores"].extend(scores)
                result["gil_interpretations"].extend(gil_interpretations)
                result["lil_interpretations"].extend(lil_interpretations)
        return result


