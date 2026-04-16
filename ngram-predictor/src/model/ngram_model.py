
import os
import json
from collections import Counter, defaultdict
import itertools

def read_env(env_path):
    env = {}
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    k, v = line.strip().split('=', 1)
                    env[k.strip()] = v.strip()
    return env

class NGramModel:
    """
    Builds, stores, and exposes n-gram probability tables and backoff lookup across all orders from 1 up to NGRAM_ORDER.
    Handles vocabulary construction, n-gram counting, MLE probability computation, and backoff logic for inference.
    """
    def __init__(self, ngram_order=4, unk_threshold=1):
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.vocab_list = []
        self.model = {f"{i}gram": dict() for i in range(1, ngram_order+1)}

    def build_vocab(self, token_file):
        """
        Build vocabulary from tokenized sentences, applying UNK_THRESHOLD.
        Parameters:
            token_file (str): Path to tokenized sentences file.
        Returns:
            None. Sets self.vocab and self.vocab_list.
        """
        word_counts = Counter()
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                word_counts.update(tokens)
        self.vocab = set([w for w, c in word_counts.items() if c > self.unk_threshold])
        self.vocab.add('<UNK>')
        self.vocab_list = sorted(self.vocab)

    def build_counts_and_probabilities(self, token_file):
        """
        Count all n-grams at orders 1 through NGRAM_ORDER and compute MLE probabilities.
        Parameters:
            token_file (str): Path to tokenized sentences file.
        Returns:
            None. Populates self.model with probability tables.
        """
        ngram_counts = {i: Counter() for i in range(1, self.ngram_order+1)}
        context_counts = {i: Counter() for i in range(2, self.ngram_order+1)}
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = [t if t in self.vocab else '<UNK>' for t in line.strip().split()]
                for n in range(1, self.ngram_order+1):
                    for i in range(len(tokens)-n+1):
                        ngram = tuple(tokens[i:i+n])
                        ngram_counts[n][ngram] += 1
                        if n > 1:
                            context = tuple(tokens[i:i+n-1])
                            context_counts[n][context] += 1
        # 1-gram probabilities
        total_1grams = sum(ngram_counts[1].values())
        self.model['1gram'] = {w[0]: c/total_1grams for w, c in ngram_counts[1].items()}
        # Higher order probabilities
        for n in range(2, self.ngram_order+1):
            table = defaultdict(dict)
            for ngram, count in ngram_counts[n].items():
                context = ' '.join(ngram[:-1])
                word = ngram[-1]
                context_count = context_counts[n][ngram[:-1]]
                if context_count > 0:
                    table[context][word] = count / context_count
            self.model[f'{n}gram'] = dict(table)

    def lookup(self, context):
        """
        Backoff lookup: try the highest-order context first, fall back to lower orders down to 1-gram.
        Parameters:
            context (list of str): List of previous tokens (length <= NGRAM_ORDER-1)
        Returns:
            dict: {word: probability} from the highest order that matches, or empty dict if no match.
        """
        for n in range(self.ngram_order, 1, -1):
            if len(context) >= n-1:
                ctx = ' '.join(context[-(n-1):])
                table = self.model.get(f'{n}gram', {})
                if ctx in table:
                    return table[ctx]
        # 1-gram fallback
        if '1gram' in self.model:
            return self.model['1gram']
        return {}

    def save_model(self, model_path):
        """
        Save all probability tables to model.json.
        Parameters:
            model_path (str): Path to save model.json
        Returns:
            None
        """
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(self.model, f, indent=2)

    def save_vocab(self, vocab_path):
        """
        Save vocabulary list to vocab.json.
        Parameters:
            vocab_path (str): Path to save vocab.json
        Returns:
            None
        """
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab_list, f, indent=2)

    def load(self, model_path, vocab_path):
        """
        Load model.json and vocab.json into the instance.
        Parameters:
            model_path (str): Path to model.json
            vocab_path (str): Path to vocab.json
        Returns:
            None
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            self.model = json.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab_list = json.load(f)
            self.vocab = set(self.vocab_list)

def main():
    import argparse
    env = read_env(os.path.join(os.path.dirname(__file__), '../../config/.env'))
    ngram_order = int(env.get('NGRAM_ORDER', 4))
    unk_threshold = int(env.get('UNK_THRESHOLD', 1))
    parser = argparse.ArgumentParser(description='NGramModel utility for ngram-predictor')
    parser.add_argument('--token_file', type=str, required=True, help='Tokenized sentences file')
    parser.add_argument('--model_path', type=str, required=True, help='Output path for model.json')
    parser.add_argument('--vocab_path', type=str, required=True, help='Output path for vocab.json')
    args = parser.parse_args()

    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)
    model.build_vocab(args.token_file)
    model.build_counts_and_probabilities(args.token_file)
    model.save_model(args.model_path)
    model.save_vocab(args.vocab_path)
    print(f"Model and vocab saved. NGRAM_ORDER={ngram_order}, UNK_THRESHOLD={unk_threshold}")

if __name__ == "__main__":
    main()
#RUN Example (build model from train_tokens.txt and save to model/model.json and model/vocab.json)
#python ngram-predictor/src/model/ngram_model.py --token_file ngram-predictor/data/processed/train_tokens.txt --model_path ngram-predictor/data/model/model.json --vocab_path ngram-predictor/data/model/vocab.json