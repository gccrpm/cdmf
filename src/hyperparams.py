# data path
DATA = 'ml-10m'
DATA_PATH = '../dataset/using-data/' + DATA
FNAMES = {'movies': 'cleaned_movies.csv',
          'actors': 'cleaned_actors.csv',
          'summaries': 'cleaned_summaries.csv',
          'storylines': 'cleaned_storylines.csv',
          'train_ratings': 'train_ratings.csv',
          'dev_ratings': 'dev_ratings.csv',
          'eval_ratings': 'eval_ratings.csv',
          'all_actors': 'all_{}main_actors.csv',
          'all_ratings': 'cleaned_ratings.csv',
          'bu': 'users_bias.csv',
          'bm': 'movies_bias.csv'}

# training scheme
BATCH_SIZE = 256
NUM_EPOCHS = 20
EVAL_EVERY = 3000

# model params
FORCED_SEQ_LEN = 195  # max sentences length is 195
VOCAB_SIZE = 8000 # 8000
PLAY_TIMES = 2 # 2
DIM_LANTENT = 50 # 100
DIM_HIDDEN1 = 200 # 300
DIM_HIDDEN2 = 50 # 100
# NUM_MOST_INFO = 9 # 9
NUM_MAIN_ACTORS = 10 #10

# network params
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 128 # 128
FEATURE_SIZE = 50 # 50
FILTERS_SIZE_LIST = [3, 4, 5] # [3,4,5]
NUM_FILTERS = 100 # 100
DROPOUT_KEEP_PROB = 0.2 # 0.2

L2_REG_LAMBDA_U = 0.2 # 0.02
L2_REG_LAMBDA_M = 0.02 # 0.02
L2_REG_LAMBDA_CNN = 0.02 # 0.02
L2_REG_LAMBDA_INFO1 = 0.01 # 0.01
L2_REG_LAMBDA_INFO2 = 0.01 # 0.01
L2_REG_LAMBDA_ACTORS1 = 0.01 # 0.01
L2_REG_LAMBDA_ACTORS2 = 0.01 # 0.01

NOISE_RATE = 0

