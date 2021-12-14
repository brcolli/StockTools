from TwitterSentimentModel import *


def load_model() -> SentimentModelLearning:
    sentiment_model_params = SentimentModelParameters(epochs=10,
                                                      batch_size=128,
                                                      load_model=False,
                                                      checkpoint_model=True,
                                                      saved_model_bin='../data/analysis/Model Results/Saved Models/'
                                                                      'best_sentiment_model.h5')

    sentiment_model_data = SentimentModelData(nsc=NSC(), base_data_csv='../data/Learning Data/'
                                                                       'sentiment_learning.csv',
                                              test_size=0.1)

    sentiment_model_learning = SentimentModelLearning(sentiment_model_params, sentiment_model_data)
    sentiment_model_learning.build_model()

    return sentiment_model_learning
