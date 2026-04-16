
import os
import sys
import argparse
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/data_prep')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/inference')))
from normalizer import Normalizer
from ngram_model import NGramModel
from predictor import Predictor

def run_dataprep():
    input_folder = 'ngram-predictor/data/raw/train'
    output_file = 'ngram-predictor/data/processed/train_tokens.txt'
    norm = Normalizer()
    texts = norm.load(input_folder)
    all_sentences = []
    for text in texts:
        text = norm.strip_gutenberg(text)
        text = norm.normalize(text)
        sentences = norm.sentence_tokenize(text)
        tokenized = [norm.word_tokenize(s) for s in sentences]
        all_sentences.extend(tokenized)
    norm.save(all_sentences, output_file)
    print(f"Data prep complete. Saved to {output_file}.")

def run_model():
    token_file = 'ngram-predictor/data/processed/train_tokens.txt'
    model_path = 'ngram-predictor/data/model/model.json'
    vocab_path = 'ngram-predictor/data/model/vocab.json'
    import dotenv
    env_path = os.path.join('ngram-predictor', 'config', '.env')
    env = {}
    if os.path.exists(env_path):
        for line in open(env_path, 'r', encoding='utf-8'):
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                env[k.strip()] = v.strip()
    ngram_order = int(env.get('NGRAM_ORDER', 4))
    unk_threshold = int(env.get('UNK_THRESHOLD', 1))
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)
    model.build_vocab(token_file)
    model.build_counts_and_probabilities(token_file)
    model.save_model(model_path)
    model.save_vocab(vocab_path)
    print(f"Model built and saved to {model_path} and {vocab_path}.")

def run_inference():
    model_path = 'ngram-predictor/data/model/model.json'
    vocab_path = 'ngram-predictor/data/model/vocab.json'
    ngram_order = 4
    env_path = os.path.join('ngram-predictor', 'config', '.env')
    env = {}
    if os.path.exists(env_path):
        for line in open(env_path, 'r', encoding='utf-8'):
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                env[k.strip()] = v.strip()
    if 'NGRAM_ORDER' in env:
        ngram_order = int(env['NGRAM_ORDER'])
    k = 3
    if 'TOP_K' in env:
        k = int(env['TOP_K'])
    model = NGramModel(ngram_order=ngram_order)
    model.load(model_path, vocab_path)
    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)
    print("Type a sequence of words and get top-k predictions. Type 'quit' to exit.")
    try:
        while True:
            text = input('> ')
            if text.strip().lower() in {'quit', 'exit'}:
                print('Goodbye.')
                break
            predictions = predictor.predict_next(text, k)
            print(f"Predictions: {predictions}")
    except KeyboardInterrupt:
        print('\nGoodbye.')

def main():
    parser = argparse.ArgumentParser(description='NGram Predictor CLI')
    parser.add_argument('--step', type=str, required=True, choices=['dataprep', 'model', 'inference', 'all'], help='Pipeline step to run')
    args = parser.parse_args()

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join('ngram-predictor', 'config', '.env'))
    except ImportError:
        pass

    if args.step == 'dataprep':
        run_dataprep()
    elif args.step == 'model':
        run_model()
    elif args.step == 'inference':
        run_inference()
    elif args.step == 'all':
        run_dataprep()
        run_model()
        run_inference()

if __name__ == "__main__":
    main()
#RUN Example
#python ngram-predictor/main.py --step all
#python ngram-predictor/main.py --step inference
#python ngram-predictor/main.py --step model
#python ngram-predictor/main.py --step dataprep
