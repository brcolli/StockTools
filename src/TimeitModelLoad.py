import time
import timeit

TF = '''import tensorflow as tf'''
IMPORTS = '''import NSCTokenizerLoadTest as tlt'''
# IMPORTS = '''pass'''
LOAD_DATA_CSV = '''tlt.load_from_csv()'''
LOAD_DATA_DILL = '''tlt.load_model_data_from_dill()'''
LOAD_MODEL_CSV = '''tlt.load_spam_model(tlt.load_from_csv())'''
LOAD_MODEL_DILL = '''tlt.load_spam_model(tlt.load_model_data_from_dill())'''

PREP_PREDICT_MODEL = '''
import NSCTokenizerLoadTest as tlt
model = tlt.load_spam_model(tlt.load_model_data_from_dill())
'''

PREDICT_MODEL = '''
tlt.model_predict(model)
'''

BUILD_MODEL = '''
tlt.load_spam_model(data)
'''

PREP_BUILD_MODEL = '''
import NSCTokenizerLoadTest as tlt
data = tlt.load_model_data_from_dill()
'''


n = 3


with open('timer-logs.txt', 'a') as f:
    f.write('\n')

    # f.write(f"Load Tensorflow x {n}:")
    # f.write (str((1/n)*timeit.timeit(stmt = TF,
    #                      number = n)))
    # f.write('\n')
    #
    # a = time.time()
    # from NSCTokenizerLoadTest import *
    # b = time.time() - a
    # f.write(f"Load Modules: {b}")
    #
    # f.write(f"Load Data Dill x {n}:")
    # f.write (str((1/n)*timeit.timeit(setup = IMPORTS,
    #                      stmt = LOAD_DATA_DILL,
    #                      number = n)))
    # f.write('\n')
    #
    # f.write(f"Load Model Dill x {n}:")
    # f.write (str((1/n)*timeit.timeit(setup = IMPORTS,
    #                      stmt = LOAD_MODEL_DILL,
    #                      number = n)))
    # f.write('\n')
    #
    # f.write(f"Load Data CSV x {n}:")
    # f.write (str((1/n)*timeit.timeit(setup = IMPORTS,
    #                      stmt = LOAD_DATA_CSV,
    #                      number = n)))
    # f.write('\n')
    #
    # f.write(f"Load Model CSV x {n}:")
    # f.write (str((1/n)*timeit.timeit(setup = IMPORTS,
    #                  stmt = LOAD_MODEL_CSV,
    #                  number = n)))
    # f.write('\n')

    # f.write(f"Build Model x {n}:")
    # f.write (str((1/n)*timeit.timeit(setup = PREP_BUILD_MODEL,
    #                  stmt = BUILD_MODEL,
    #                  number = n)))
    # f.write('\n')

    f.write(f"Predict Model x {n}:")
    f.write (str((1/n)*timeit.timeit(setup = PREP_PREDICT_MODEL,
                     stmt = PREDICT_MODEL,
                     number = n)))
    f.write('\n')