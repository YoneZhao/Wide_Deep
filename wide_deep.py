# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import tensorflow as tf

_USER_BUCKET_SIZE = 200000
_ITEM_BUCKET_SIZE = 100000
_PUBLISHER_BUCKET_SIZE = 100000
_SONG_BUCKET_SIZE = 20000
_ARTIST_BUCKET_SIZE = 20000

_CROSS_BUCKET_SIZE = 50000
_COUNTRY_LIST = ['AD', 'AE', 'AF', 'AI', 'AL', 'AM', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AW', 'AX', 'AZ', 'BD', 'BE', 'BG',
                 'BH', 'BN', 'BO', 'BR', 'BT', 'BY', 'BZ', 'CA', 'CH', 'CI', 'CL', 'CM', 'CN', 'CO', 'CR', 'CU', 'CV',
                 'CY', 'CZ', 'DE', 'DK', 'DO', 'DZ', 'EC', 'EG', 'ES', 'FI', 'FJ', 'FR', 'GA', 'GB', 'GE', 'GF', 'GH',
                 'GL', 'GN', 'GR', 'GT', 'HK', 'HN', 'HR', 'HT', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IO', 'IQ', 'IR',
                 'IS', 'IT', 'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KM', 'KP', 'KR', 'KW', 'KZ', 'LA', 'LB', 'LK',
                 'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'MK', 'ML', 'MM', 'MO', 'MT', 'MU', 'MV', 'MX', 'MY', 'MZ',
                 'NA', 'NG', 'NI', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PA', 'PE', 'PF', 'PH', 'PK', 'PL', 'PR', 'PS', 'PT',
                 'PW', 'PY', 'QA', 'RO', 'RS', 'RU', 'SA', 'SD', 'SE', 'SG', 'SI', 'SK', 'SN', 'SV', 'SY', 'TH', 'TN',
                 'TR', 'TT', 'TW', 'TZ', 'UA', 'UG', 'US', 'UY', 'UZ', 'VE', 'VN', 'VU', 'YE', 'ZA', 'ZM']
_LANGUAGE_LIST = ['ar', 'as', 'bho', 'bn', 'da', 'de', 'en', 'es', 'fr', 'gu', 'hi', 'hry', 'id', 'in', 'it', 'ja',
                  'kn', 'ko', 'ml', 'mr', 'ms', 'or', 'pa', 'pt', 'raj', 'ta', 'te', 'th', 'vi', 'zh']
_PROVINCE_LIST = ['AP', 'AS', 'BR', 'CH', 'CT', 'DL', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'MH', 'ML', 'MN', 'MP',
                  'NL', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TG', 'TN', 'TR', 'UP', 'UT', 'WB']
_GRADE_LIST = ['A++', 'A+', 'A', 'B', 'C', 'D']
_HAS_PROFILE_LIST = ['True', 'False']
_SONG_SRC_LIST = [-1, 0, 1, 100, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35,
                  36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 5, 50, 51, 9, 97, 99]
_USER_LEVEL_BUCKETS = 8
_GENDER_BUCKETS = 4
_AGE_BUCKETS = 6
_TYPE_BUCKETS = 9
_RECALL_LEVEL_BUCKETS = 4
_SONG_GENRES_BUCKETS = 37

_CSV_COLUMNS = [
    'label', 'user_id', 'user_country', 'user_province', 'user_level', 'user_gender', 'user_age', 'has_profile',
    'item_id', 'item_lang', 'pub_id', 'media_type', 'grade', 'recall_level', 'song_id', 'artist_id', 'song_src',
    'song_genres', 'report_num', 'share_num', 'like_num ', 'comment_num', 'duration'
]
_CSV_COLUMN_DEFAULTS = [[0], [0], [''], [''], [0], [0], [0], [''], [0], [''], [0], [0], [''], [0], [0], [0], [0], [0],
                        [0], [0], [0], [0], [0]]
_SHUFFLE_SIZE = 400000


def build_model_columns():
    """Builds a set feature columns."""
    # categorical ID feature
    user_id = tf.feature_column.categorical_column_with_hash_bucket('user_id', _USER_BUCKET_SIZE)
    item_id = tf.feature_column.categorical_column_with_hash_bucket('tiem_id', _ITEM_BUCKET_SIZE)
    pub_id = tf.feature_column.categorical_column_with_hash_bucket('pub_id', _PUBLISHER_BUCKET_SIZE)
    song_id = tf.feature_column.categorical_column_with_hash_bucket('song_id', _SONG_BUCKET_SIZE)
    artist_id = tf.feature_column.categorical_column_with_hash_bucket('artist_id', _ARTIST_BUCKET_SIZE)
    # categorical feature
    user_country = tf.feature_column.categorical_column_with_vocabulary_list('user_country', _COUNTRY_LIST,
                                                                             dtype=tf.string)
    user_province = tf.feature_column.categorical_column_with_vocabulary_list('user_province', _PROVINCE_LIST,
                                                                              dtype=tf.string)
    user_level = tf.feature_column.categorical_column_with_identity('user_level', num_buckets=_USER_LEVEL_BUCKETS)
    user_gender = tf.feature_column.categorical_column_with_identity('user_gender', num_buckets=_GENDER_BUCKETS)
    user_age = tf.feature_column.categorical_column_with_identity('user_age', num_buckets=_AGE_BUCKETS)
    has_profile = tf.feature_column.categorical_column_with_vocabulary_list('has_profile', _HAS_PROFILE_LIST)
    item_lang = tf.feature_column.categorical_column_with_vocabulary_list('item_lang', _LANGUAGE_LIST)
    grade = tf.feature_column.categorical_column_with_vocabulary_list('grade', _GRADE_LIST)
    media_type = tf.feature_column.categorical_column_with_identity('media_type', num_buckets=_TYPE_BUCKETS)
    recall_level = tf.feature_column.categorical_column_with_identity('recall_level', num_buckets=_RECALL_LEVEL_BUCKETS)
    song_src = tf.feature_column.categorical_column_with_vocabulary_list('song_src', _SONG_SRC_LIST)
    song_genres = tf.feature_column.categorical_column_with_identity('song_genres', num_buckets=_SONG_GENRES_BUCKETS)
    # numeric feature
    report_num = tf.feature_column.numeric_column('report_num')
    share_num = tf.feature_column.numeric_column('share_num')
    like_num = tf.feature_column.numeric_column('like_num')
    comment_num = tf.feature_column.numeric_column('comment_num')
    duration = tf.feature_column.numeric_column('duration')

    common_columns = [
        tf.feature_column.indicator_column(user_country),
        tf.feature_column.indicator_column(user_province),
        tf.feature_column.indicator_column(user_gender),
        tf.feature_column.indicator_column(user_age),
        tf.feature_column.indicator_column(user_level),
        tf.feature_column.indicator_column(has_profile),
        tf.feature_column.indicator_column(item_lang),
        tf.feature_column.indicator_column(grade),
        tf.feature_column.indicator_column(media_type),
        tf.feature_column.indicator_column(recall_level),
        tf.feature_column.indicator_column(song_src),
        tf.feature_column.indicator_column(song_genres),
        tf.feature_column.embedding_column(user_id, dimension=16),
        tf.feature_column.embedding_column(item_id, dimension=16),
        tf.feature_column.embedding_column(pub_id, dimension=16),
        tf.feature_column.embedding_column(song_id, dimension=10),
        tf.feature_column.embedding_column(artist_id, dimension=10)
    ]
    wide_columns = [
        user_age,
        user_level,
        has_profile,
        grade,
        recall_level,
        tf.feature_column.crossed_column(['gender', 'item_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['gender', 'song_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['gender', 'artist_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['gender', 'pub_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['age', 'item_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['age', 'song_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['age', 'artist_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['age', 'pub_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_country', 'item_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_country', 'song_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_country', 'artist_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_province', 'item_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_id', 'item_lang'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_id', 'pub_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_id', 'song_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_id', 'artist_id'], hash_bucket_size=_CROSS_BUCKET_SIZE),
        tf.feature_column.crossed_column(['user_id', 'song_genres'], hash_bucket_size=_CROSS_BUCKET_SIZE),
    ]
    wide_columns += common_columns
    # deep columns
    deep_columns = [
        report_num,
        share_num,
        like_num,
        comment_num,
        duration,
    ]
    deep_columns += common_columns
    return wide_columns, deep_columns


def input_fn_common(data_path, shuffle, num_epochs=10, batch_size=1024 * 5):
    """Generate an input function for the Estimator."""

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        return features, labels

    dataset = tf.data.TextLineDataset(data_path)
    if shuffle:
        dataset = dataset.shuffle(_SHUFFLE_SIZE)
    dataset = dataset.map(parse_csv, num_parallel_calls=10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def input_fn_train():



wide_columns, deep_columns = build_model_columns()

estimator = tf.estimator.DNNLinearCombinedClassifier(linear_feature_columns=wide_columns,
                                                     linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.01),
                                                     dnn_feature_columns=deep_columns,
                                                     dnn_hidden_units=[256, 128, 64],
                                                     dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

estimator.train(input_fn=)
