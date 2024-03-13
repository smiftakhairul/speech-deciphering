import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class StatsVisualizer:
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
        
    def predictor_confusion_matrix(self, y_true, y_pred, languages):
        cm = confusion_matrix(y_true, y_pred, labels=languages)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=languages, yticklabels=languages)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    def predictor_classification_report(self, report):
        if report:
            report_data = []
            lines = report.split('\n')
            for line in lines[2:-3]:
                row_data = line.split()
                if len(row_data) == 5:
                    row = {}
                    row['class'] = row_data[0]
                    row['precision'] = float(row_data[1])
                    row['recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    # row['support'] = int(row_data[4])
                    report_data.append(row)

            df = pd.DataFrame.from_dict(report_data)
            df.set_index('class', inplace=True)
            df.plot(kind='bar', figsize=(10, 6))
            plt.title('Classification Report')
            plt.xlabel('Class')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()
