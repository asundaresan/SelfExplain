import logging
import json
from collections import OrderedDict, Counter

import tqdm
import spacy
import torch
from transformers import AutoTokenizer, AutoModel, RobertaConfig, XLNetConfig
from transformers.modeling_utils import SequenceSummary

from .utils import chunks

config_dict = {'xlnet-base-cased': XLNetConfig,
               'roberta-base': RobertaConfig}

def concept_store(model_name, input_file_name, output_folder, max_concept_length, batch_size=8, use_sentence=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')
    model.eval()
    config = config_dict[model_name]
    sequence_summary = SequenceSummary(config)

    # initialize spacy 
    nlp = spacy.load("en_core_web_sm")

    concept_set = set()

    total = 0
    with open(input_file_name, 'r') as input_file:
        total = sum(1 for line in input_file)
    print(f"found {total} lines in {input_file_name}")

    logging.info(f"loading input from {input_file_name}")
    sentence_lengths = Counter()
    with open(input_file_name, 'r') as input_file:
        for i, line in enumerate(tqdm.tqdm(input_file, total=total, desc="loading concepts")):
            json_line = json.loads(line)
            if use_sentence:
                # sentence is actually the full text
                text = json_line["sentence"].strip().strip(' .')
                doc = nlp(text)
                for sentence in doc.sents:
                    sentence = sentence.text
                    sentence_length = len(sentence.split())
                    sentence_lengths.update([sentence_length,])

                    if len(sentence.split()) <= max_concept_length:
                        concept_set.add()
            else:
                phrase_labels = ["NP", "VP",]
                for leaf in json_line["parse_tree"]:
                    if leaf["phrase_label"] in phrase_labels:
                        phrase = leaf["phrase"].lower()
                        phrase_len = len(phrase.split())
                        if phrase_len < max_concept_length:
                            concept_set.add(phrase)


    concept_idx = {i: value for i, value in enumerate(concept_set)}
    print(f"mapped {total} inputs to {len(concept_set)} concepts.")

    num_batches = len(concept_idx)//batch_size
    concept_tensor = []
    for batch in tqdm.tqdm(chunks(list(concept_idx.values()), n=batch_size), desc="building concepts", total=num_batches):
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        for key, value in inputs.items():
            inputs[key] = value.to('cuda')
        outputs = model(**inputs)
        pooled_rep = sequence_summary(outputs[0])
        concept_tensor.append(pooled_rep.detach().cpu())

    concept_tensor = torch.cat(concept_tensor, dim=0)

    filename = f'{output_folder}/concept_store.pt'
    print(f"saving concept_tensor in {filename}")
    torch.save(concept_tensor, filename)

    with open(f'{output_folder}/concept_idx.json', 'w') as out_file:
        json.dump(concept_idx, out_file, indent=2)

    return

