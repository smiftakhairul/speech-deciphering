import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from stats_visualizer import StatsVisualizer
import joblib

class LanguagePredictor:
    def __init__(self, data_file, model_file):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.vectorizer = None
        self.languages = []
        self.visualizer = StatsVisualizer(self.data_file)
    
    def train(self):
        data = pd.read_csv(self.data_file)
        self.languages = [col for col in data.columns if col != 'Sentence']
        
        all_sentences = []
        labels = []
        for lang in self.languages:
            all_sentences.extend(data[lang].tolist())
            labels.extend([lang] * len(data))
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(all_sentences, labels, test_size=0.2, random_state=42)
        self.vectorizer = TfidfVectorizer()
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        self.model = Pipeline([('tfidf', self.vectorizer), ('clf', clf)])
        
        # Train and Evaluate the model
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        classified_report = classification_report(y_test, y_pred)
        print("Model accuracy:", acc)
        print("Classification Report:")
        print(classified_report)
        self.visualizer.predictor_confusion_matrix(y_test, y_pred, self.languages)
        self.visualizer.predictor_classification_report(classified_report)
    
    def predict(self, sentence):
        predicted_language = self.model.predict([sentence])[0]
        return predicted_language
    
    def save_model(self):
        joblib.dump((self.model, self.vectorizer, self.languages), self.model_file)
        print("Model saved successfully.")

    def load_model(self):
        self.model, self.vectorizer, self.languages = joblib.load(self.model_file)
        print("Model loaded successfully.")
