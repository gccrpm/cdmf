import os
import logging
import itertools

import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

from data_utils.text_util import clean_str, pad_sentence, build_vocab
from data_utils.noise_uitl import generate_random_noise


class DataLoad(object):
    logging.getLogger().setLevel(logging.INFO)

    def __init__(self, data_path, fnames, forced_seq_len, vocab_size, paly_times,
                 num_main_actors, batch_size, num_epochs, noise_rate):
        self.data_path = data_path
        self.fnames = fnames
        self.forced_seq_len = forced_seq_len
        self.vocab_size = vocab_size
        self.paly_times = paly_times
        self.num_main_actors = num_main_actors
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.noise_rate = noise_rate

        # data file path
        self.info_path = os.path.join(data_path, fnames['movies'])
        self.actors_path = os.path.join(data_path, fnames['actors'])
        self.summaries_path = os.path.join(data_path, fnames['summaries'])
        self.storylines_path = os.path.join(data_path, fnames['storylines'])
        self.all_ratings_path = os.path.join(data_path, fnames['all_ratings'])
        self.all_actors_path = os.path.join(
            data_path, fnames['all_actors'].format(self.num_main_actors))
        self.users_bias = os.path.join(data_path, fnames['bu'])
        self.movies_bias = os.path.join(data_path, fnames['bm'])

        # generate features
        self._generate_id_mappings()
        self._generate_bias()
        self._generate_info()  # dim = M*self.dim_onehot
        self._generate_actors()  # dim = M*self.dim_onehot
        self._generate_descriptions()  # dim = M*self.forced_seq_len

    def load_data(self, mode, num_sub=100*100):
        if mode in ('train', 'dev', 'eval'):
            ratings_path = os.path.join(
                self.data_path, self.fnames['{}_ratings'.format(mode)])
        else:
            raise ValueError('please choose correct mode (train/dev/eval)')
        
        df_ratings = pd.read_csv(
            ratings_path, 
            header=0, 
            dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float, 'timestamp':np.str})
        
        if mode == 'train':
            return self._train_batch_iterator(df_ratings)
        else:
            return self._dev_eval_iterator(df_ratings, num_sub=num_sub)

    def _train_batch_iterator(self, df):
        num_batches_per_epoch = df.shape[0] // self.batch_size + 1
        # shuffle trian dataset
        df = df.sample(frac=1).reset_index(drop=True)
        # generate train batch
        for i in range(num_batches_per_epoch):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, df.shape[0])
            batch_df = df.iloc[start_idx:end_idx]

            batch_uids = list(batch_df.loc[:, 'userId'])
            batch_mids = list(batch_df.loc[:, 'movieId'])
            batch_bu_seq = np.array([self.bu_dict[u] for u in batch_uids], np.float)
            batch_bm_seq = np.array([self.bm_dict[m] for m in batch_mids], np.float)
            
            batch_u_oids = np.array([self.uid2order[u] for u in batch_uids])
            batch_m_oids = np.array([self.mid2order[m] for m in batch_mids])
            batch_info = self.info_mat[batch_m_oids, :]
            batch_actors = self.actors_mat[batch_m_oids, :]
            batch_descriptions = self.descriptions_mat[batch_m_oids, :]

            batch_u_oids = np.reshape(batch_u_oids, (batch_u_oids.shape[0], 1))
            batch_m_oids = np.reshape(batch_m_oids, (batch_m_oids.shape[0], 1))

            X_user = (batch_u_oids, batch_bu_seq)
            X_movie = (batch_m_oids, batch_info, batch_actors, batch_descriptions, batch_bm_seq)
            
            X = (X_user, X_movie)
            Y = np.array(batch_df.loc[:, 'rating'])
            yield X, Y

    def _dev_eval_iterator(self, df, num_sub):
        # shuffle dev dataset
        df = df.sample(frac=1).reset_index(drop=True)
        num_sub_per_eval = df.shape[0] // num_sub + 1
        # generate sub df for each dev
        for i in range(num_sub_per_eval):
            start_idx = i * num_sub
            end_idx = min((i+1) * num_sub, df.shape[0])
            sub_df = df.iloc[start_idx:end_idx]

            sub_uids = list(sub_df.loc[:, 'userId'])
            sub_mids = list(sub_df.loc[:, 'movieId'])
            sub_bu_seq = np.array([self.bu_dict[u] for u in sub_uids], np.float)
            sub_bm_seq = np.array([self.bm_dict[m] for m in sub_mids], np.float)
        
            sub_u_oids = np.array([self.uid2order[u] for u in sub_uids])
            sub_m_oids = np.array([self.mid2order[m] for m in sub_mids])
            sub_info = self.info_mat[sub_m_oids, :]
            sub_actors = self.actors_mat[sub_m_oids, :]
            sub_descriptions = self.descriptions_mat[sub_m_oids, :]

            sub_u_oids = np.reshape(sub_u_oids, (sub_u_oids.shape[0], 1))
            sub_m_oids = np.reshape(sub_m_oids, (sub_m_oids.shape[0], 1))

            X_user = (sub_u_oids, sub_bu_seq)
            X_movie = (sub_m_oids, sub_info, sub_actors, sub_descriptions, sub_bm_seq)

            X = (X_user, X_movie)
            Y = np.array(sub_df.loc[:, 'rating'])
            yield X, Y

    def _generate_id_mappings(self):
        df_all_ratings = pd.read_csv(self.all_ratings_path, header=0,
                                     dtype={'userId': np.int32, 'movieId': np.int32,
                                            'rating': np.float, 'timestamp': np.str})
        all_uids = sorted(list(df_all_ratings.loc[:, 'userId'].unique()))
        all_mids = sorted(list(df_all_ratings.loc[:, 'movieId'].unique()))
        self.num_all_users = len(all_uids)
        self.num_all_movies = len(all_mids)
        self.uid2order = {u: i for i, u in enumerate(all_uids)}
        self.order2uid = {i: u for i, u in enumerate(all_uids)}
        self.mid2order = {m: i for i, m in enumerate(all_mids)}
        self.order2mid = {i: m for i, m in enumerate(all_mids)}
        self.mu = np.mean(df_all_ratings.loc[:, 'rating'])

        return self

    def _generate_bias(self):
        df_users_bias = pd.read_csv(
            self.users_bias, header=0, dtype={'userId':np.int32, 'bias':np.float})
        df_movies_bias = pd.read_csv(
            self.movies_bias, header=0, dtype={'movieId':np.int32, 'bias':np.float})
        uids = list(df_users_bias.loc[:, 'userId'])
        mids = list(df_movies_bias.loc[:, 'movieId'])
        bu_seq = list(df_users_bias.loc[:, 'bias'])
        bm_seq = list(df_movies_bias.loc[:, 'bias'])

        self.bu_dict = dict(zip(uids, bu_seq))
        self.bm_dict = dict(zip(mids, bm_seq))

        return self

    def _generate_info(self):
        def _generate_year_range(year):
            if year <= 1950:
                year = 1950
            elif year <= 1960:
                year = 1960
            elif year <= 1970:
                year = 1970
            elif year <= 1980:
                year = 1980
            elif year <= 1990:
                year = 1990
            return year
        df_info = pd.read_csv(self.info_path, header=0,
                              dtype={'movieId': np.int32, 'title': np.str,
                                     'genres': np.str, 'year': np.int32})
        df_info.loc[:, 'year'] = df_info.loc[:, 'year'].apply(_generate_year_range)
        df_info.loc[:, 'genres'] = df_info.loc[:, 'genres'].apply(lambda x: x.split('|'))
        years = list(df_info.loc[:, 'year'])
        genres = list(df_info.loc[:, 'genres'])

        # build info vocabulary
        all_info = list(set(years).union(set(itertools.chain(*genres))))
        all_info += ['<OOI>']  # out of info
        self.num_all_info = len(all_info)
        info2id = {info: i for i, info in enumerate(all_info)}

        # merge year into genres
        info_list = genres.copy()
        for i in range(len(years)):
            info_list[i].append(years[i])

        self.num_most_info = max([len(info) for info in info_list])

        new_info_list = []
        for info in info_list:
            new_info = []
            for i in range(self.num_most_info):
                try:
                    new_info.append(info2id[info[i]])
                except IndexError:
                    new_info.append(info2id['<OOI>'])
            new_info_list.append(new_info)

        # dimension = N * self.dim_onehot
        self.info_mat = np.array(new_info_list)
        print('have generated feature matrix, shape={}'.format(self.info_mat.shape))
        return self

    def _generate_actors(self):
        # read all actors' name
        df_all_actors = pd.read_csv(self.all_actors_path, header=0,
                                    dtype={'name': np.str, 'times': np.int32})
        # build actors vocabulary
        selected_actors = list(
            df_all_actors.loc[df_all_actors['times'] >= self.paly_times, 'name'])
        selected_actors += ['<OTA>']  # other actors
        self.num_all_main_actors = len(selected_actors)
        actor2id = {a: i for i, a in enumerate(selected_actors)}

        # read actors for each movie
        df_actors = pd.read_csv(self.actors_path, header=0, dtype={
                                'movieId': np.int32, 'actors': np.str})

        df_actors.loc[:, 'actors'] = df_actors.loc[:, 'actors'].apply(
            lambda x: x.split('|'))
        actors_list = list(df_actors.loc[:, 'actors'])

        new_actors_list = []
        for actors in actors_list:
            new_actors = []
            for i in range(self.num_main_actors):
                try:
                    new_actors.append(actor2id[actors[i]])
                except IndexError:
                    new_actors.append(actor2id['<OTA>'])
                except KeyError:
                    new_actors.append(actor2id['<OTA>'])
            new_actors_list.append(new_actors)

        self.actors_mat = np.array(new_actors_list)
        print('have generated actor matrix, shape={}'.format(self.actors_mat.shape))
        return self

    def _generate_descriptions(self):
        df_info = pd.read_csv(self.info_path, header=0,
                              dtype={'movieId': np.int32, 'title': np.str,
                                     'genres': np.str, 'year': np.str})
        df_summaries = pd.read_csv(self.summaries_path, header=0,
                                   dtype={'movieId': np.int32, 'summary': np.str})
        df_storylines = pd.read_csv(self.storylines_path, header=0,
                                    dtype={'movieId': np.int32, 'storyline': np.str})

        titles = list(df_info.loc[:, 'title'])
        summaries = list(df_summaries.loc[:, 'summary'])
        storylines = list(df_storylines.loc[:, 'storyline'])

        porter_stemmer = PorterStemmer()
        stop = stopwords.words('english')

        raw_descriptions = [clean_str('{} {} {}'.format(t, su, st))
                            for t, su, st in zip(titles, summaries, storylines)]

        tokenized_descriptions = [[porter_stemmer.stem(word) for word in sent.split(' ') if word not in stop]
                                  for sent in raw_descriptions]

        noised_descriptions = generate_random_noise(tokenized_descriptions, self.noise_rate)

        padded_descriptions = pad_sentence(
            noised_descriptions, self.forced_seq_len)
        token2id, _, self.vocab_size = build_vocab(
            padded_descriptions, self.vocab_size)

        descriptions = []
        for sent in padded_descriptions:
            description = []
            for word in sent:
                if word not in token2id:
                    word = '<OOV>'
                description.append(token2id[word])
            descriptions.append(description)

        self.descriptions_mat = np.array(descriptions)
        print('have generated description, shape={}'.format(
            self.descriptions_mat.shape))
        return self
