import os
from dataset_generator import DatasetGenerator
from language_predictor import LanguagePredictor
from stats_visualizer import DatasetVisualizer
from utils import Utils

class MainApplication:
    def __init__(self):
        self.languages = {'bn': 'Bengali', 'ru': 'Russian', 'kk': 'Kazakh', 'es': 'Spanish'}
        self.data_file = './files/languages.csv'
        self.model_file = './models/language_predictor_model.pkl'
        self.utils = Utils()
        self.dataset = DatasetGenerator(self.languages)
        self.predictor = LanguagePredictor(self.data_file, self.model_file)
        self.visualizer = DatasetVisualizer(self.data_file)

    def generate_dataset(self, num_rows):
        df = self.dataset.generate_and_translate(num_rows)
        df.to_csv(self.data_file, index=False)
        print("CSV file generated successfully!")
    
    def visualize_dataset(self):
        self.visualizer.dataset_histogram()
        self.visualizer.dataset_language_distribution()
        
    def train_language_model(self):
        if not os.path.exists(self.model_file):
            self.predictor.train()
            self.predictor.save_model()
        else:
            self.predictor.load_model()
            print("Model already trained and loaded.")

    def predict_language(self, sentences):
        for sentence in sentences:
            predicted_language = self.predictor.predict(sentence)
            print(f"Sentence: '{sentence}' --> Predicted language: {predicted_language}")

if __name__ == "__main__":
    main_app = MainApplication()
    
    # Generate and visualize dataset
    if not os.path.exists(main_app.data_file):
        main_app.generate_dataset(5)
    main_app.visualizer.load_dataset()
    main_app.visualize_dataset()
        
    # Train or load model and predict language
    main_app.train_language_model()
    main_app.predict_language(main_app.utils.test_sentences())
