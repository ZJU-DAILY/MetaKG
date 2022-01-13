import pandas as pd
import gzip
import numpy as np
import json
import tqdm
import random
import collections
import time
random.seed(2020)

def extract_ui_rating_amazon(path):
    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    temp_user_dict = collections.defaultdict(int)
    temp_item_dict = collections.defaultdict(int)
    for ori_id, remap_id in user_dict.items():
        temp_user_dict[ori_id] = int(remap_id)+1
    for ori_id, remap_id in item_dict.items():
        temp_item_dict[ori_id] = int(remap_id)+1

    rating_list_ui = collections.defaultdict(list)
    g = gzip.open(path + 'rawdata/reviews_Books_5.json.gz', 'r')
    for idx, l in enumerate(g):
        l = eval(l)
        if temp_user_dict[l['reviewerID']]!=0 and temp_item_dict[l['asin']]!=0:
            u_id = int(user_dict[l['reviewerID']])
            i_id = int(item_dict[l['asin']])
            rating = int(l['overall'])
            rating_list_ui[u_id].append([i_id, rating])

        if idx%100000==0:
            print('idx: ', idx)

    with open(path+'/test_scenario/rating_list_ui.json', 'w') as f:
        json.dump(rating_list_ui, f)

def extract_ui_rating_yelp(path):
    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    temp_user_dict = collections.defaultdict(int)
    temp_item_dict = collections.defaultdict(int)
    for ori_id, remap_id in user_dict.items():
        temp_user_dict[ori_id] = int(remap_id) + 1
    for ori_id, remap_id in item_dict.items():
        temp_item_dict[ori_id] = int(remap_id) + 1

    rating_list_ui = collections.defaultdict(list)
    num = 0
    with open(path+'rawdata/yelp_academic_dataset_review.json', 'r') as f:
        for line in f:
            tmp = json.loads(line)
            if temp_user_dict[tmp['user_id']] != 0 and temp_item_dict[tmp['business_id']] != 0:
                num+=1
                u_id = int(user_dict[tmp['user_id']])
                i_id = int(item_dict[tmp['business_id']])
                rating = int(tmp['stars'])
                rating_list_ui[u_id].append([i_id, rating])

                if num%100000==0:
                    print(num)

    with open(path+'/test_scenario/rating_list_ui.json', 'w') as f:
        json.dump(rating_list_ui, f)


def first_reach_amazon(path):
    """
    the first timestamp user and item reached
    {user: unixtime}->{str: int}
    """
    first_reach_user = {}
    first_reach_item = {}
    user_item_interaction = {}
    item_user_interaction = {}

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')
    g = gzip.open(path+'rawdata/reviews_Books_5.json.gz', 'r')

    for idx, l in enumerate(g):
        l = eval(l)

        if l['reviewerID'] in list(user_dict.keys()):
            if l['reviewerID'] not in list(first_reach_user.keys()):
                first_reach_user[l['reviewerID']] = l['unixReviewTime']
            else:
                if l['unixReviewTime'] < first_reach_user[l['reviewerID']]:
                    first_reach_user[l['reviewerID']] = l['unixReviewTime']

        if l['asin'] in list(item_dict.keys()):
            if l['asin'] not in list(first_reach_item.keys()):
                first_reach_item[l['asin']] = l['unixReviewTime']
            else:
                if l['unixReviewTime'] < first_reach_item[l['asin']]:
                    first_reach_item[l['asin']] = l['unixReviewTime']
        if idx%1000000==0:
            print(l['reviewerID'], l['asin'], l['unixReviewTime'])
            print('idx: ', idx)


    print('user_dict: ', len(list(user_dict.keys())))
    print('first_reach_user: ', len(list(first_reach_user.keys())))
    print('item_dict: ', len(list(item_dict.keys())))
    print('first_reach_item: ', len(list(first_reach_item.keys())))

    with open(path+'first_reach_user.json', 'w') as f:
        json.dump(first_reach_user, f)
    with open(path+'first_reach_item.json', 'w') as f:
        json.dump(first_reach_item, f)

def first_reach_lfm(path):
    """
    the first timestamp user and item reached
    {user: unixtime}->{str: int}
    """
    first_reach_user = {}
    first_reach_item = {}

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    user_lines = open(path + 'rawdata/LFM-1b_users.txt', 'r').readlines()

    u_unixtime = {}
    for idx, line in enumerate(user_lines):
        if idx==0:
            continue
        l = line.strip()
        tmp = l.split()
        u_unixtime[tmp[0]] = tmp[-1]
    for key, val in user_dict.items():
        first_reach_user[key] = int(u_unixtime[key])

    lfm_LEs = open(path + 'rawdata/LFM-1b_LEs.txt', 'r').readlines()
    track_time = collections.defaultdict(int)
    for idx, line in enumerate(lfm_LEs):
        l = line.strip()
        tmp = l.split()
        track_id = int(tmp[-2])
        unixtime = int(tmp[-1])

        if track_time[track_id] == 0:
            track_time[track_id] = unixtime
        elif unixtime < track_time[track_id]:
            track_time[track_id] = unixtime

        if idx%10000000==0:
            # print('track: ', type(track_id), track_id, 'unixtime: ', type(unixtime), unixtime)
            print('idx: ', idx)

    del lfm_LEs

    item_ids = [int(it) for it in list(item_dict.keys())]
    loss_item = 0
    for it_id in item_ids:
        if track_time[it_id] == 0:
            loss_item+=1
        first_reach_item[it_id] = track_time[it_id]
    print('loss_item_num: ', loss_item)

    print('user_dict: ', len(list(user_dict.keys())))
    print('first_reach_user: ', len(list(first_reach_user.keys())))
    print('item_dict: ', len(list(item_dict.keys())))
    print('first_reach_item: ', len(list(first_reach_item.keys())))
    
    with open(path + 'first_reach_user.json', 'w') as f:
        json.dump(first_reach_user, f)
    with open(path + 'first_reach_item.json', 'w') as f:
        json.dump(first_reach_item, f)
    

def first_reach_yelp(path):
    """
    the first timestamp user and item reached
    {user/item: unixtime}->{str: int}
    """
    first_reach_user = {}
    first_reach_item = {}

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    print('start collect user_time...')
    user_time = collections.defaultdict(int)
    with open(path+'rawdata/yelp_academic_dataset_user.json', 'r') as f:
        for line in f:
            tmp = json.loads(line)
            # print(tmp)
            # print(type(tmp))
            # print(type(tmp['user_id']), tmp['user_id'], type(tmp['yelping_since']), tmp['yelping_since'])
            user_id = tmp['user_id']
            unixtime = time.strptime(tmp['yelping_since'], "%Y-%m-%d %H:%M:%S")
            unixtime = int(time.mktime(unixtime))
            user_time[user_id] = unixtime

    no_user = 0
    for key, val in user_dict.items():
        if user_time[key] == 0:
            no_user+=1
        else:
            first_reach_user[key] = user_time[key]
    print('no_user_num: ', no_user)

    print('start collect business_time...')
    business_time = collections.defaultdict(int)
    with open(path+'rawdata/yelp_academic_dataset_review.json', 'r') as f:
        for line in f:
            tmp = json.loads(line)
            # print(tmp)
            # print(type(tmp))
            # print(type(tmp['business_id']),tmp['business_id'], type(tmp['date']),tmp['date'])
            business_id = tmp['business_id']
            unixtime = time.strptime(tmp['date'], "%Y-%m-%d %H:%M:%S")
            unixtime = int(time.mktime(unixtime))

            if business_time[business_id] == 0:
                business_time[business_id] = unixtime
            elif unixtime < business_time[business_id]:
                business_time[business_id] = unixtime

    no_item = 0
    for key, val in item_dict.items():
        if business_time[key] == 0:
            no_item+=1
        else:
            first_reach_item[key] = business_time[key]
    print('no_item_num: ', no_item)

    print('user_dict: ', len(list(user_dict.keys())))
    print('first_reach_user: ', len(list(first_reach_user.keys())))
    print('item_dict: ', len(list(item_dict.keys())))
    print('first_reach_item: ', len(list(first_reach_item.keys())))

    with open(path + 'first_reach_user.json', 'w') as f:
        json.dump(first_reach_user, f)
    with open(path + 'first_reach_item.json', 'w') as f:
        json.dump(first_reach_item, f)

def read_user_list(path):
    """
    return: dict{org_id: remap_id} type: {str: str}
    """
    lines = open(path, 'r').readlines()
    user_dict = dict()
    for idx, line in enumerate(lines):
        if idx==0:
            continue
        l = line.strip()
        tmp = l.split()
        user_dict[tmp[0]] = tmp[1]

    return user_dict

def read_item_list(path):
    """
    return: dict{org_id: remap_id}   type: {str: str}
    """
    lines = open(path, 'r').readlines()
    item_dict = dict()
    for idx, line in enumerate(lines):
        if idx==0:
            continue
        l = line.strip()
        tmp = l.split()
        # item_dict[tmp[0]] = tmp[1]
        item_dict[tmp[0]] = str(idx-1)

    return item_dict

def merge_train_vali_test(path):
    """
    return: entire {users: items}
    """
    user_dict = dict()
    lines_train = open(path+'train.txt', 'r').readlines()
    lines_vali = open(path+'valid1.txt', 'r').readlines()
    lines_test = open(path + 'test.txt', 'r').readlines()
    for l_train, l_vali, l_test in zip(lines_train, lines_vali, lines_test):
        tmp_train = l_train.strip()
        tmp_vali = l_vali.strip()
        tmp_test = l_test.strip()
        inter_train = [int(i) for i in tmp_train.split()]
        inter_vali = [int(i) for i in tmp_vali.split()]
        inter_test = [int(i) for i in tmp_test.split()]

        user_id_train, item_ids_train = inter_train[0], inter_train[1:]
        user_id_vali, item_ids_vali = inter_vali[0], inter_vali[1:]
        user_id_test, item_ids_test = inter_test[0], inter_test[1:]
        item_ids_train = set(item_ids_train)
        item_ids_vali = set(item_ids_vali)
        item_ids_test = set(item_ids_test)

        item_ids_merge = item_ids_train | item_ids_vali | item_ids_test
        user_dict[user_id_train] = list(item_ids_merge)

    with open(path+'user_items_all.json', 'w') as f:
        json.dump(user_dict, f)

def contruct_test_scenario(path):
    """
    return: test_scenario
    """
    with open(path+'first_reach_user.json', 'r') as f:
        first_reach_user = json.load(f)
    with open(path+'first_reach_item.json', 'r') as f:
        first_reach_item = json.load(f)

    # sorted by timestamp
    user_timestamp = sorted(first_reach_user.items(), key=lambda x: x[1])
    item_timestamp = sorted(first_reach_item.items(), key=lambda x: x[1])

    print(len(user_timestamp), len(item_timestamp))
    print('type_user: ', type(user_timestamp[0][0]), 'type_timestamp: ', type(user_timestamp[0][1]))

    # (org_id, timestamp)  exist:new == 8:2
    new_user = user_timestamp[int(0.8*len(user_timestamp)):]
    exist_user = user_timestamp[:int(0.8 * len(user_timestamp))]
    new_item = item_timestamp[int(0.8*len(item_timestamp)):]
    exist_item = item_timestamp[:int(0.8 * len(item_timestamp))]

    print(len(new_user), len(exist_user), len(new_item), len(exist_item))

    user_dict = read_user_list(path + 'user_list.txt')
    item_dict = read_item_list(path + 'item_list.txt')

    # get remap_id of user or item
    new_user = [int(user_dict[t[0]]) for t in new_user]
    exist_user = [int(user_dict[t[0]]) for t in exist_user]
    new_item = [int(item_dict[t[0]]) for t in new_item]
    exist_item = [int(item_dict[t[0]]) for t in exist_item]

    print(new_user[:5])
    print(new_item[:5])

    # construct the test_scenario
    meta_training = dict()
    warm_up = dict()
    user_cold = dict()
    item_cold = dict()
    user_item_cold = dict()

    with open(path+'user_items_all.json', 'r') as f:
        user_item_all = json.load(f)
    for key, value in user_item_all.items():
        if int(key) in new_user:
            user_cold[int(key)] = list(set(value) & set(exist_item))
            user_item_cold[int(key)] = list(set(value) & set(new_item))
        elif int(key) in exist_user:
            item_cold[int(key)] = list(set(value) & set(new_item))
            meta_training[int(key)] = list(set(value) & set(exist_item))

    for i in range(int(0.1*len(exist_user))):
        idx = random.sample(meta_training.keys(), 1)[0]
        # print(idx, meta_training[idx])
        warm_up[idx] = meta_training[idx]
        del meta_training[idx]

    with open(path+'test_scenario/'+'meta_training.json', 'w') as f:
        json.dump(meta_training, f)
    with open(path+'test_scenario/'+'warm_up.json', 'w') as f:
        json.dump(warm_up, f)
    with open(path+'test_scenario/'+'user_cold.json', 'w') as f:
        json.dump(user_cold, f)
    with open(path+'test_scenario/'+'item_cold.json', 'w') as f:
        json.dump(item_cold, f)
    with open(path+'test_scenario/'+'user_item_cold.json', 'w') as f:
        json.dump(user_item_cold, f)

def support_query_set(path):
    path_test = path+'test_scenario/'
    for s in state:
        path_json = path_test + s + '.json'
        with open(path_json, 'r') as f:
            scenario = json.load(f)
        support_txt = open(path_test + s + '_support.txt', mode='w')
        query_txt = open(path_test + s + '_query.txt', mode='w')

        for u,i in scenario.items():
            if len(i)>=13 and len(i)<=100:
                random.shuffle(i)
                support = i[:-10]
                query = i[-10:]
                support_txt.write(u)
                query_txt.write(u)
                for s_one in support:
                    support_txt.write(' '+str(s_one))
                for q_one in query:
                    query_txt.write(' '+str(q_one))
                support_txt.write('\n')
                query_txt.write('\n')
        support_txt.close()
        query_txt.close()

if __name__ == '__main__':
    state = ['meta_training', 'warm_up', 'user_cold', 'item_cold', 'user_item_cold']

    dataset = 'last-fm' # 'amazon-book', 'last-fm', 'yelp2018'

    if dataset == 'amazon-book':
        path = './datasets/amazon-book/'
        first_reach_amazon(path)
    elif dataset == 'last-fm':
        path = './datasets/last-fm/'
        first_reach_lfm(path)
    elif dataset == 'yelp2018':
        path = './datasets/yelp2018/'
        first_reach_yelp(path)
    
    merge_train_vali_test(path)
    contruct_test_scenario(path)
    support_query_set(path)

    # extract_ui_rating_amazon(path)
    # extract_ui_rating_yelp(path)
