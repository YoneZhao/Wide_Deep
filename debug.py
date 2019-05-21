import tensorflow as tf

pets = {'pets': ['rabbit', 'pig', 'dog', 'mouse', 'cat', '']}
pets = {'pets': [-1, 0, 1, 5, 7, 8]}
# 1: list
column = tf.feature_column.categorical_column_with_vocabulary_list(
    key='pets',
    vocabulary_list=['cat', 'dog', 'rabbit', 'pig'],
    dtype=tf.string, num_oov_buckets=1)
# default_value=-1,
# num_oov_buckets=3)
# 2. identity
level_feats = {'level': [2, 3, 5, 1]}
level = tf.feature_column.categorical_column_with_identity(key='level', num_buckets=8)

# 3. list int
column = tf.feature_column.categorical_column_with_vocabulary_list('pets', [-1, 0, 1, 7], dtype=tf.int32, num_oov_buckets=0)

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(pets, [indicator])
# indicator = tf.feature_column.indicator_column(level)
# tensor = tf.feature_column.input_layer(level_feats, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))



