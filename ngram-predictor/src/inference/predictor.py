
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_prep')))
from ngram_model import NGramModel
from normalizer import Normalizer

class Predictor:
    """
    Accepts a pre-loaded NGramModel and Normalizer via the constructor, normalizes input text, and returns the top-k predicted next words sorted by probability. Backoff lookup is delegated to NGramModel.lookup().
    """
    def __init__(self, model, normalizer):
        """
        Initialize Predictor with pre-loaded NGramModel and Normalizer.
        Parameters:
            model (NGramModel): Pre-loaded NGramModel instance
            normalizer (Normalizer): Pre-loaded Normalizer instance
        """
        self.model = model
        self.normalizer = normalizer
        self.ngram_order = model.ngram_order
        self.vocab = set(model.vocab_list)

    def normalize(self, text):
        """
        Normalize input text and extract last NGRAM_ORDER-1 words as context.
        Parameters:
            text (str): Input text string
        Returns:
            list: List of last NGRAM_ORDER-1 normalized tokens
        """
        norm_text = self.normalizer.normalize(text)
        tokens = norm_text.split()
        context = tokens[-(self.ngram_order-1):] if len(tokens) >= self.ngram_order-1 else tokens
        return context

    def map_oov(self, context):
        """
        Replace out-of-vocabulary words in context with <UNK>.
        Parameters:
            context (list): List of tokens
        Returns:
            list: List with OOV words replaced by <UNK>
        """
        return [w if w in self.vocab else '<UNK>' for w in context]

    def predict_next(self, text, k=3):
        """
        Predict top-k next words given input text.
        Parameters:
            text (str): Input text string
            k (int): Number of top predictions to return
        Returns:
            list: Top-k predicted next words sorted by probability
        """
        context = self.normalize(text)
        context = self.map_oov(context)
        candidates = self.model.lookup(context)
        if not candidates:
            return []
        sorted_words = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:k]]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predictor utility for ngram-predictor')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model.json')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocab.json')
    parser.add_argument('--text', type=str, required=True, help='Input text string')
    parser.add_argument('--k', type=int, default=3, help='Number of top predictions to return')
    args = parser.parse_args()

    model = NGramModel()
    model.load(args.model_path, args.vocab_path)
    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)
    predictions = predictor.predict_next(args.text, args.k)
    print(predictions)

if __name__ == "__main__":
    main()
#RUN Example (return the top-3 predictions)
#python ngram-predictor/src/inference/predictor.py --model_path ngram-predictor/data/model/model.json --vocab_path ngram-predictor/data/model/vocab.json --text "the adventure of" --k 3