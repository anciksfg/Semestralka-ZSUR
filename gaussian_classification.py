import matplotlib.pyplot as plt
import numpy as np
import math
import time


def load(nazev):
    """
    vstup:
        * nazev - nazev souboru
    vystup:
        * seznam dat (mx2)
    """
    c1 = []
    c2 = []
    with open(nazev, 'r') as f:
        for line in f:
            line = line.split()
            (a, b) = float(line[0]), float(line[1])
            c1.append(a)
            c2.append(b)
    return np.column_stack((c1, c2))


# GaussClf will be the cals that will have the Gaussian naive bayes classifier implementation
class GaussClf:
    def separete_by_classes(self, X, y):
        """
        Separates our dataset in subdatasets by classes
        :param X:
        :param y:
        :return:
        """
        classes_index = {}
        subdatasets = {}
        cls, counts = np.unique(y, return_counts=True)
        self.classes = cls
        self.class_freq = dict(zip(cls, counts))
        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(y == class_type)
            subdatasets[class_type] = X[classes_index[class_type], :]
            self.class_freq[class_type] = self.class_freq[class_type] / sum(list(self.class_freq.values()))
        return subdatasets

    def fit(self, X, y):
        """
        The fitting function
        :param X:
        :param y:
        :return:
        """
        separated_X = self.separete_by_classes(X, y)
        self.means = {}
        self.std = {}
        for class_type in self.classes:
            # Calculate the mean and the standart deviation from datasets
            self.means[class_type] = np.mean(separated_X[class_type], axis=0)[0]
            self.std[class_type] = np.std(separated_X[class_type], axis=0)[0]

    def calculate_probability(self, x, mean, stdev):
        """
        Calculates the class probability using gaussian distribution
        :param x:
        :param mean:
        :param stdev:
        :return:
        """
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev ** 2)) * exponent

    def predict_proba(self, X):
        """
        Predicts the probability for every class
        :param X:
        :return:
        """
        self.class_prob = {cls:math.log(self.class_freq[cls], math.e) for cls in self.classes}
        for cls in self.classes:
            for i in range(len(self.means)):
                print(X[i])

            self.class_prob[cls] += self.calculate_probability(X[i], self.means[cls][i], self.std[cls][i])
            self.class_prob = {cls: math.e**self.class_prob[cls] for cls in self.class_prob}
        return self.class_prob

    def predict(self, X):
        """
        Predicts the class of a sample
        :param X:
        :return:
        """
        pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for cls, prob in self.predict_proba(X).items():
                if prob > max_prob:
                    max_prob = prob
                    pred_class = cls
            pred.append(pred_class)
        return pred

# testovaci cast souboru
if __name__ == '__main__':
    # Načtení dat
    X = load('data.txt')

    indexes = np.random.choice(X.shape[0], replace=False, size=1000)
    Vyber = X[indexes, :].copy()
