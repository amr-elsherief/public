# Normalizer class
#%%
#%%

import os
import re

class Normalizer:
    def load(self, folder_path):
        """Load all .txt files from a folder and return their contents as a list of strings."""
        texts = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        return texts

    def strip_gutenberg(self, text):
        """Remove Gutenberg header and footer from text."""
        # Gutenberg header/footer markers
        start_re = r'\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*'
        end_re = r'\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*'
        start_match = re.search(start_re, text, re.IGNORECASE)
        end_match = re.search(end_re, text, re.IGNORECASE)
        start = start_match.end() if start_match else 0
        end = end_match.start() if end_match else len(text)
        return text[start:end].strip()

    def lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        import string
        # Add extra unicode punctuation to string.punctuation
        extra_punct = '“”‘’—–'
        all_punct = string.punctuation + extra_punct
        return text.translate(str.maketrans('', '', all_punct))

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_whitespace(self, text):
        # Remove extra spaces and blank lines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def normalize(self, text):
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        # Simple sentence tokenizer (split on . ! ?)
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if s.strip()]

    def word_tokenize(self, sentence):
        # Split on whitespace
        return sentence.split()

    def save(self, sentences, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                if isinstance(sentence, list):
                    f.write(' '.join(sentence) + '\n')
                else:
                    f.write(str(sentence) + '\n')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Normalizer utility for ngram-predictor')
    parser.add_argument('--input_folder', type=str, help='Folder with .txt files to process')
    parser.add_argument('--output_file', type=str, help='Output file for tokenized sentences')
    args = parser.parse_args()

    norm = Normalizer()
    if args.input_folder and args.output_file:
        texts = norm.load(args.input_folder)
        all_sentences = []
        for text in texts:
            text = norm.strip_gutenberg(text)
            text = norm.normalize(text)
            sentences = norm.sentence_tokenize(text)
            tokenized = [norm.word_tokenize(s) for s in sentences]
            all_sentences.extend(tokenized)
        norm.save(all_sentences, args.output_file)
        print(f"Processed {len(all_sentences)} sentences. Saved to {args.output_file}.")
    else:
        print("Please provide --input_folder and --output_file arguments.")

if __name__ == "__main__":
    main()
#RUN Example (process all .txt files in data/raw and save tokenized sentences to data/processed/train_tokens.txt)
#python ngram-predictor/src/data_prep/normalizer.py --input_folder ngram-predictor/data/raw/train --output_file ngram-predictor/data/processed/train_tokens.txt