
import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_prep')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../inference')))
from ngram_model import NGramModel
from normalizer import Normalizer
from predictor import Predictor


class PredictorUI:
    """
    Provides a browser-based interface for n-gram prediction using Streamlit.
    Allows ngram order selection, input folder change, pipeline reload, and auto-predict.
    """
    def __init__(self):
        self.model = None
        self.normalizer = None
        self.predictor = None
        self.k = 3
        self.ngram_order = 4
        self.model_path = 'ngram-predictor/data/model/model.json'
        self.vocab_path = 'ngram-predictor/data/model/vocab.json'
        self.input_folder = 'ngram-predictor/data/raw/train'
        self.token_file = 'ngram-predictor/data/processed/train_tokens.txt'

    def reload_pipeline(self):
        # Data prep
        norm = Normalizer()
        texts = norm.load(self.input_folder)
        all_sentences = []
        for text in texts:
            text = norm.strip_gutenberg(text)
            text = norm.normalize(text)
            sentences = norm.sentence_tokenize(text)
            tokenized = [norm.word_tokenize(s) for s in sentences]
            all_sentences.extend(tokenized)
        norm.save(all_sentences, self.token_file)
        # Model
        model = NGramModel(ngram_order=self.ngram_order)
        model.build_vocab(self.token_file)
        model.build_counts_and_probabilities(self.token_file)
        model.save_model(self.model_path)
        model.save_vocab(self.vocab_path)
        self.model = model
        self.normalizer = norm
        self.predictor = Predictor(self.model, self.normalizer)

    def load_model(self):
        self.model = NGramModel(ngram_order=self.ngram_order)
        self.model.load(self.model_path, self.vocab_path)
        self.normalizer = Normalizer()
        self.predictor = Predictor(self.model, self.normalizer)

    def run(self):
        st.title("NGram Predictor UI")
        st.write("Enter a sequence of words to get top-k next word predictions.")
        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            self.input_folder = st.text_input("Input folder train material", self.input_folder, key="input_folder_text")
            self.ngram_order = st.selectbox("NGram order", options=[1,2,3,4], index=self.ngram_order-1)
            self.k = st.number_input("Top-k words", min_value=1, max_value=10, value=self.k)
            auto_predict = st.checkbox("Auto-predict (no click needed)", value=True)
            if st.button("Reload/Refresh Pipeline"):
                self.reload_pipeline()
                st.success("Pipeline reloaded: data prep and model rebuilt.")
            if st.button("Load Model Only"):
                self.load_model()
                st.success("Model loaded from disk.")

        # Ensure model is loaded
        if self.model is None or self.predictor is None:
            try:
                self.load_model()
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return

        # Use session state for live prediction
        if 'input_text' not in st.session_state:
            st.session_state['input_text'] = ''
        if 'predictions' not in st.session_state:
            st.session_state['predictions'] = []

        def update_prediction():
            text = st.session_state['input_text']
            if text:
                st.session_state['predictions'] = self.predictor.predict_next(text, int(self.k))
            else:
                st.session_state['predictions'] = []

        text = st.text_input("Input text", value=st.session_state['input_text'], key="input_text", on_change=update_prediction if auto_predict else None)

        predictions = st.session_state['predictions']
        if not auto_predict and st.button("Predict"):
            update_prediction()
            predictions = st.session_state['predictions']
        if predictions:
            st.write(f"Predictions: {predictions}")

def main():
    ui = PredictorUI()
    ui.run()

if __name__ == "__main__":
    main()

# RUN Example
#pip install streamlit
#python -m streamlit run ngram-predictor/src/ui/app.py
#python -m streamlit run ngram-predictor/src/ui/app.py -- --model_path ngram-predictor/data/model/model.json --vocab_path ngram-predictor/data/model/vocab.json --ngram_order 4 --k 3
