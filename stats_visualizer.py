import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetVisualizer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.dataset_df = None
    
    def load_dataset(self):
        self.dataset_df = pd.read_csv(self.data_file)

    def dataset_histogram(self):
        plt.figure(figsize=(10, 6))
        for column in self.dataset_df.columns:
            self.dataset_df[column].apply(lambda x: len(x.split())).hist(alpha=0.5, bins=20, label=column)

        plt.title('Distribution of Sentence Lengths')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def dataset_language_distribution(self):
        color_palette = sns.color_palette("husl", len(self.dataset_df.columns))
        lang_counts = self.dataset_df.count()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=lang_counts.index, y=lang_counts.values, hue=lang_counts.index, palette=color_palette, legend=False)
        plt.title('Number of Sentences Translated to Each Language')
        plt.xlabel('Language')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
