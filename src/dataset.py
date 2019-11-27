import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, dataset_file_location):
        self.dataset_file_location = dataset_file_location
        self._load_dataset_from_file()
        self._generate_training_and_test()

    def _load_dataset_from_file(self):
        self.main_dataset = pd.read_csv(self.dataset_file_location, header=None, names=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
            'thai', 'num'], index_col=False)

    def draw_distribution(self, column):
        t0_dataset = self.main_dataset.loc[self.main_dataset['num'] == 0].loc[:, column]
        t1_dataset = self.main_dataset.loc[self.main_dataset['num'] == 1].loc[:, column]

        sns.distplot(a=t0_dataset, hist=True, rug=True, label='0')
        sns.distplot(a=t1_dataset, hist=True, rug=True, label='1')
        plt.show()

    def _generate_training_and_test(self):
        pass
