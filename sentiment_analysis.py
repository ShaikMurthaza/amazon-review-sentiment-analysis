import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import datetime
import re

class SentimentAnalyzer:
    def __init__(self, output_dir='sentiment_analysis_results'):
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'model')
        self.tokenizer_dir = os.path.join(output_dir, 'tokenizer')
        self.results_dir = os.path.join(output_dir, 'results')
        
        for directory in [self.output_dir, self.model_dir, self.tokenizer_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)

    def preprocess_text(self, text):
        """Preprocess review text with improved cleaning"""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        text = ' '.join(text.split())
        return text

    def load_and_preprocess_data(self, parsed_reviews):
        df = pd.DataFrame(parsed_reviews)
        df['full_text'] = df.apply(
            lambda row: f"{str(row.get('title', ''))} {str(row.get('text', ''))}".strip(), 
            axis=1
        )
        df['processed_text'] = df['full_text'].apply(self.preprocess_text)
        df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4.0 else 0)
        
        return df


    def prepare_model_input(self, df, max_words=10000, max_len=200):
        """Prepare input with increased vocabulary size"""
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(df['processed_text'])
        
        X = tokenizer.texts_to_sequences(df['processed_text'])
        X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
        
        y = df['sentiment'].values
        
        return X, y, tokenizer

    def build_sentiment_model(self, max_words, max_len, embedding_dim=100):
        """Enhanced model architecture"""
        model = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            LSTM(128, return_sequences=True),
            Dropout(0.5),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model


    def train_and_evaluate(self, parsed_reviews):
        df = self.load_and_preprocess_data(parsed_reviews)
        X, y, tokenizer = self.prepare_model_input(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = self.build_sentiment_model(10000, 200)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'dataset_info': {
                'total_reviews': len(df),
                'positive_reviews': int(df['sentiment'].sum()),
                'negative_reviews': len(df) - int(df['sentiment'].sum()),
                'train_size': len(X_train),
                'test_size': len(X_test)
            },
            'model_performance': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'training_history': {
                    'accuracy': [float(x) for x in history.history['accuracy']],
                    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                }
            },
            'confusion_matrix': conf_matrix.tolist()
        }
        
        model_path = os.path.join(self.model_dir, 'sentiment_model.keras')
        model.save(model_path, save_format='keras')
        
        tokenizer_path = os.path.join(self.tokenizer_dir, 'tokenizer.json')
        tokenizer_json = tokenizer.to_json()
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(tokenizer_json)
        
        results_path = os.path.join(self.results_dir, 'model_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        return results

    def predict_sentiment(self, text, model_path, tokenizer_path):
        """Fixed prediction method with proper tokenizer loading"""
        model = load_model(model_path)
        
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
        
        processed_text = self.preprocess_text(text)
        
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        
        prediction = float(model.predict(padded)[0][0])
        confidence = max(prediction, 1 - prediction)
        
        return {
            'review_text': text,
            'processed_text': processed_text,
            'sentiment': 'Positive' if prediction > 0.5 else 'Negative',
            'confidence': confidence,
            'raw_score': prediction
        }

def main():
    csv_path = 'sentiment\dataset\parsed_reviews.csv'
    
    try:
        df = pd.read_csv(csv_path, nrows=30000, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, nrows=30000, encoding='latin1')
    
    parsed_reviews = df.to_dict('records')
    
    analyzer = SentimentAnalyzer()
    
    try:
        results = analyzer.train_and_evaluate(parsed_reviews)
        print("Model training completed successfully!")
        print(f"Model accuracy: {results['model_performance']['accuracy']:.2%}")
        
        sample_reviews = [
            "Great product, works perfectly!",
            "Terrible experience, would not recommend.",
            "Decent product, but could be improved."
        ]
        
        model_path = os.path.join(analyzer.model_dir, 'sentiment_model.keras')
        tokenizer_path = os.path.join(analyzer.tokenizer_dir, 'tokenizer.json')
        
        print("\nSample predictions:")
        for review in sample_reviews:
            pred = analyzer.predict_sentiment(review, model_path, tokenizer_path)
            print(f"\nReview: {review}")
            print(f"Sentiment: {pred['sentiment']} (confidence: {pred['confidence']:.2%})")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()