import torch
from tqdm import tqdm
from sys import argv
import json
import re


class ParaphraseModel:
    def __init__(self):
        self.de2en = torch.hub.load(
            'pytorch/fairseq',
            'transformer.wmt19.de-en',
            checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
            tokenizer='moses',
            bpe='fastbpe')
        self.en2de = torch.hub.load(
            'pytorch/fairseq',
            'transformer.wmt19.en-de',
            checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
            tokenizer='moses',
            bpe='fastbpe')
    def paraphrases(self, english_sentence, number):
        german = self.en2de.translate(english_sentence)
        enc = self.de2en.encode(german)
        return [self.de2en.decode(x['tokens']) for x in self.de2en.generate(enc, number)]

    def paraphrase_data(self, example, count):
        phrases = [example['question']] + list(self.paraphrases(example['question'], count))
        categorized = {}
        for paraphrase in phrases:
            tok = normalize(paraphrase)
            if tok in categorized:
                continue
            categorized[tok] = paraphrase

        for paraphrase in categorized.values():
            copy = dict(example)
            copy['question'] = paraphrase
            copy['question_toks'] = tokenize(paraphrase)
            copy['human_wording'] = example['question']
            yield copy
    def paraphrase_dataset(self, dataset, count):
        return [para for eg in tqdm(dataset) for para in self.paraphrase_data(eg, count)]

def normalize(sentence):
    return tuple(re.sub("[^a-zA-Z0-9]", " ", sentence).lower().split())

def tokenize(s):
    """
    Approximate tokenization
    """
    s = re.sub("([\(\)\[\],?.;:])", r" \1 ", s)
    toks = s.split()
    full_toks = []
    while toks:
        tok = toks.pop(0)
        if len(tok) > 2 and tok[-1] in ".,?')]":
            toks = [tok[:-1], tok[-1]] + toks
            continue
        elif len(tok) > 1 and tok[0] == '"':
            toks = ["``", tok[1:]] + toks
        elif len(tok) > 1 and tok[-1] == '"':
            toks = [tok[:-1], "''"] + toks
        elif len(tok) > 1 and tok[0] in "([":
            toks = [tok[0], tok[1:]] + toks
        elif len(tok) > 2 and tok[-2:] == "'s":
            toks = [tok[:-2], tok[-2:]] + toks
        else:
            full_toks.append(tok)
    return full_toks

if __name__ == '__main__':
    _, input_file, output_file, count = argv
    with open(input_file) as f:
        dataset = json.load(f)
    model = ParaphraseModel()
    paraphrased_dataset = model.paraphrase_dataset(dataset, int(count))
    with open(output_file, "w") as f:
        json.dump(paraphrased_dataset, f, indent=4)