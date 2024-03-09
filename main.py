import os
from dataset_generator import DatasetGenerator

class MainApplication:
    def __init__(self):
        self.dataset = DatasetGenerator()

    def generate_dataset(self, num_rows):
        df = self.dataset.generate_and_translate(num_rows)
        df.to_csv('./files/languages.csv', index=False)
        print("CSV file generated successfully!")

if __name__ == "__main__":
    main_app = MainApplication()
    
    # Generate Dataset
    if not os.path.exists('./files/languages.csv'):
        main_app.generate_dataset(2)
    else:
        print("languages.csv already exists. Skipping dataset generation.")
