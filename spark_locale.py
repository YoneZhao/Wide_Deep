from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import pyspark
from datetime import datetime, timedelta
import time
import sys
import json
from pyspark import SparkContext
from operator import add


PATH = '/Users/Zhao/Data/'
feature_path = PATH + 'online_feature.log'
res_path = PATH + 'track_res.log'
save_path = PATH + 'train_data'


def data_parser(x, isFeature=False):
    pos = str(x).find('{')
    if pos >= 0:
        x = str(x)[pos:].strip()
        raw_dict = json.loads(x)
        if isFeature is False:
            rec = {'user_id': raw_dict['user_id'], 'item_id': raw_dict['item_id'],
                   'action': raw_dict['action'], 'timestamp': raw_dict['timestamp'],
                   'duration': raw_dict['duration'], 'play_time': raw_dict['play_time']}
            if len(rec) == 6:
                return rec
            return None
        rec = {'UserID': raw_dict['UserID'], 'ItemID': raw_dict['ItemID'], 'TimeStamp': raw_dict['TimeStamp'],
               'UserInfo': raw_dict['UserInfo'], 'ItemInfo': raw_dict['ItemInfo']}
        return rec


def res_data_transfer(x):
    label = 0
    if x['action'] not in ['show', 'unwanted']:
        label = 1
    if x['action'] == 'click':
        return None
    if x['action'] == 'duration':
        duration = float(x['duration'])
        playtime = float(x['play_time'])
        if playtime / duration < 0.1:
            label = 0
    return '%d|%d' % (int(x['user_id']), int(x['item_id'])), (label, x['timestamp'], x['action'])


def res_compare(x, y):
    if x[0] == y[0]:
        if x[2] != 'duration' and y[2] != 'duration':
            return x[1] > y[1]
        return x[2] != 'duration'
    return x[0] > y[0]


def feat_transfer(x):
    return '%d|%d' % (int(x['UserID']), int(x['ItemID'])), (x['TimeStamp'], x['UserInfo'], x['ItemInfo'])


def join_mapper(x):
    (_, (x, y)) = x
    return x[1] > y[0]


def csv_mapper(data):
    """
    将join后数据转为csv格式
    :param data:
    :return:
    """
    (_, ((label, res_timestamp, action), (feat_timestamp, user_info, item_info))) = data
    record_info = item_info.get('recording', '')
    if record_info == '':
        return
    song_info = record_info.get('song_info', '')
    if song_info == '':
        return
    # USER PART
    user_id = user_info.get('user_id', 0)
    if user_id == 0:
        return
    user_country = user_info.get('country', '')
    user_province = user_info.get('province', '')
    user_level = user_info.get('user_level', 0)
    user_gender = user_info.get('gender', 0)
    user_age = user_info.get('age', 0)
    has_profile = user_info.get('has_profile_image', False)
    # ITEM PART
    item_id = item_info.get('sm_id', 0)
    if item_id == 0:
        return
    item_lang = item_info.get('language', '').split('-')[0]
    pub_id = item_info.get('user_id', '')
    report_num = item_info.get('report_num', 0)
    share_num = item_info.get('share_num', 0)
    like_num = item_info.get('like_num', 0)
    comment_num = item_info.get('comment_num', 0)
    # RECORDING
    media_type = record_info.get('media_type', 0)
    duration = record_info.get('duration', 0)
    grade = record_info.get('grade', '')
    recall_level = record_info.get('recall_level', 0)
    # SONG
    song_id = song_info.get('song_id', 0)
    artist_id = song_info.get('artist_id', 0)
    song_src = song_info.get('song_src', 0)
    song_genres = song_info.get('song_genres', [])
    s = ''
    feats = [label]
    feats += [user_id, user_country, user_province, user_level, user_gender, user_age, has_profile]
    feats += [item_id, item_lang, pub_id, media_type, grade, recall_level, song_id, artist_id, song_src]
    if len(song_genres) >= 1:
        feats.append(song_genres[0])
    else:
        feats.append(36)
    feats += [report_num, share_num, like_num, comment_num, duration]
    feats = map(str, feats)
    s = ','.join(feats)
    return user_id, (label, s)


def mapper1(input):
    (uid, ((label, data), counter)) = input
    return data


def get_data(path, dt, length, partitions=100, isFeature=False):
    y, m, d, h = dt[0:4]
    ret = sc.parallelize([])
    # 合并
    for i in range(length):
        try:
            ndt = datetime(y, m, d, h) - timedelta(hours=i)
            date = ndt.strftime('%Y%m%d')
            dateh = ndt.strftime('%Y%m%d%H')
            data = sc.textFile(path % (date, dateh)).repartition(partitions).map(lambda x: data_parser(x, isFeature)).filter(
                lambda x: x is not None)
            ret = ret.union(data)
        except:
            pass
    return ret



'''
user_id, user_country, user_province, user_level, user_gender, user_age, has_profile, item_id, item_lang, pub_id, 
media_type, grade, recall_level, song_id, artist_id, song_src, song_genres, report_num, share_num, like_num, 
comment_num, duration, label
'''
sc = SparkContext()
res_data = sc.textFile(res_path).repartition(10).map(lambda x: data_parser(x)).filter(lambda x: x is not None)
# {'user_id': 2251799814268211, 'item_id': 147306977, 'action': 'show', 'timestamp': 1555905600, 'duration': 220,
# 'play_time': 1}
res_data = res_data.map(res_data_transfer).filter(lambda x: x is not None). \
    reduceByKey(lambda x, y: x if res_compare(x, y) else y)
# ('2251799814268211|213301284', (0, 1555905600, 'show'))

feat_data = sc.textFile(feature_path).repartition(10).map(lambda x: data_parser(x, True)).filter(
    lambda x: x is not None)
feat_data = feat_data.map(feat_transfer).filter(lambda x: x is not None)
train_data = res_data.join(feat_data).filter(join_mapper).reduceByKey(lambda x, y: x if x[1][0] > y[1][0] else y). \
    filter(lambda x: x is not None)
train_data = train_data.map(csv_mapper).filter(lambda x: x is not None)
print(train_data.first())
# pos_users = train_data.map(lambda x: (x[0], x[1][0])).reduceByKey(add).filter(lambda x: x[1] > 0)
# train_data = train_data.join(pos_users).map(mapper1).filter(lambda x: x is not None)
# print("train_size:", train_data.count())
# train_data.repartition(1).saveAsTextFile(path=save_path, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
#
