from TwitterModelInterface import *


def load_from_csv():
    nsc = NSC()

    test_size = 0.1
    aug_data_csv = None
    save_preload_binary = ''
    from_preload_binary = ''
    base_data_csv = '../data/Learning Data/spam_learning.csv'

    features_to_train = ['full_text', 'cap.english', 'cap.universal',
                                 'raw_scores.english.overall',
                                 'raw_scores.universal.overall',
                                 'raw_scores.english.astroturf',
                                 'raw_scores.english.fake_follower',
                                 'raw_scores.english.financial',
                                 'raw_scores.english.other',
                                 'raw_scores.english.self_declared',
                                 'raw_scores.english.spammer',
                                 'raw_scores.universal.astroturf',
                                 'raw_scores.universal.fake_follower',
                                 'raw_scores.universal.financial',
                                 'raw_scores.universal.other',
                                 'raw_scores.universal.self_declared',
                                 'raw_scores.universal.spammer',
                                 'favorite_count', 'retweet_count']

    spam_model_data = SpamModelData(nsc=nsc, base_data_csv=base_data_csv,
                                    test_size=test_size,
                                    features_to_train=features_to_train,
                                    aug_data_csv=aug_data_csv, save_preload_binary=save_preload_binary,
                                    from_preload_binary=from_preload_binary)

    return spam_model_data


def save_to_dill(model_data: SpamModelData):
    data = (model_data.text_input_length, model_data.nsc.tokenizer)
    with open('../data/Learning Data/SaveTests/nsc.dill', 'wb') as mb:
        dill.dump(data, mb)


def load_from_dill():
    with open('../data/Learning Data/SaveTests/nsc.dill', 'rb') as mb:
        data = dill.load(mb)

    return data


def load_model_data_from_dill():
    t_input_len, tk = load_from_dill()
    nsc = NSC()
    nsc.tokenizer = tk
    test_size = 0.1
    features_to_train = ['full_text', 'cap.english', 'cap.universal',
                         'raw_scores.english.overall',
                         'raw_scores.universal.overall',
                         'raw_scores.english.astroturf',
                         'raw_scores.english.fake_follower',
                         'raw_scores.english.financial',
                         'raw_scores.english.other',
                         'raw_scores.english.self_declared',
                         'raw_scores.english.spammer',
                         'raw_scores.universal.astroturf',
                         'raw_scores.universal.fake_follower',
                         'raw_scores.universal.financial',
                         'raw_scores.universal.other',
                         'raw_scores.universal.self_declared',
                         'raw_scores.universal.spammer',
                         'favorite_count', 'retweet_count']

    spam_model_data = SpamModelData(nsc, '', test_size, features_to_train=features_to_train, load_nsc_only=True,
                                    text_input_length=t_input_len)
    return spam_model_data


def load_spam_model(model_data):

    spam_model_params = SpamModelParameters(load_model=True,
                                            saved_model_bin='../data/analysis/Model Results/Saved '
                                                            'Models/best_spam_model.h5',
                                            evaluate_model=False)
    spam_model_learning = SpamModelLearning(spam_model_params, model_data)
    spam_model_learning.build_model()
    # x = spam_model_learning.predict(csv='../data/Learning Data/spam_test.csv')

    return spam_model_learning


def model_predict(model_learning):
    return model_learning.predict(csv='../data/Learning Data/spam_test.csv')


