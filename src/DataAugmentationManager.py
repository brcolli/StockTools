from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE


class DataAugmentationManager:

    def __init__(self):
        return

    def smote_for_classification(self, x, y):

        # values to evaluate
        k_values = [1, 2, 3, 4, 5, 6, 7]
        for k in k_values:

            # define pipeline
            model = DecisionTreeClassifier()
            over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
            under = RandomUnderSampler(sampling_strategy=0.5)
            steps = [('over', over), ('under', under), ('model', model)]
            pipeline = Pipeline(steps=steps)

            # evaluate pipeline
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, x, y, scoring='roc_auc', cv=cv, n_jobs=-1)
            return mean(scores)

    def borderline_smote(self, x, y):

        # summarize class distribution
        counter = Counter(y)
        print(counter)

        # transform the dataset
        oversample = BorderlineSMOTE()
        x, y = oversample.fit_resample(x, y)

        # summarize the new class distribution
        return Counter(y)

    def adaptive_synthetic_sampling(self, x, y):

        # summarize class distribution
        counter = Counter(y)
        print(counter)

        # transform the dataset
        oversample = ADASYN()
        x, y = oversample.fit_resample(x, y)

        # summarize the new class distribution
        return Counter(y)


def main():
    dam = DataAugmentationManager


if __name__ == '__main__':
    main()
