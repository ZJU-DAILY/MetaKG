import os
import random
import logging
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from model.MetaKG import MetaKG
from utility.parser_Metakg import *
from utility.log_helper import *
from utility.metrics import *
from utility.helper import *
from utility.DataLoader_MetaKG import DataLoader


def evaluate(model, train_graph, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
    model.eval()

    with torch.no_grad():
        att = model.compute_attention(train_graph)
    train_graph.edata['att'] = att

    n_users = len(test_user_dict.keys())
    item_ids_batch = item_ids.cpu().numpy()

    cf_scores = []
    precision = []
    recall = []
    ndcg = []

    with torch.no_grad():
        for user_ids_batch in user_ids_batches:
            cf_scores_batch = model('predict', train_graph, user_ids_batch, item_ids)       # (n_batch_users, n_eval_items)

            cf_scores_batch = cf_scores_batch.cpu()
            user_ids_batch = user_ids_batch.cpu().numpy()
            precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(cf_scores_batch, train_user_dict, test_user_dict, user_ids_batch, item_ids_batch, K)

            cf_scores.append(cf_scores_batch.numpy())
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)

    cf_scores = np.concatenate(cf_scores, axis=0)
    precision_k = sum(np.concatenate(precision)) / n_users
    recall_k = sum(np.concatenate(recall)) / n_users
    ndcg_k = sum(np.concatenate(ndcg)) / n_users
    return cf_scores, precision_k, recall_k, ndcg_k


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoader(args, logging, state='meta_training')
    cold_data = DataLoader(args, logging, state='user_item_cold')


    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # evaluation user_id_batches
    user_ids = list(cold_data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]  # (n, 10000)
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # construct model & optimizer
    model = MetaKG(args, data.n_users, data.n_entities, data.n_relations, user_pre_embed, item_pre_embed)

    # continue to training the stored model
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    # if n_gpu > 1:
    #     model = nn.parallel.DistributedDataParallel(model)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # move graph data to GPU
    train_graph = data.train_graph
    train_graph = train_graph.to(device)

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    precision_list10 = []
    recall_list10 = []
    ndcg_list10 = []
    precision_list20 = []
    recall_list20 = []
    ndcg_list20 = []

    epoch = 1
    # meta_training
    model.train()

    # train cf
    time1 = time()
    cf_total_loss = 0

    '''
    train_user_list = []
    for key, value in data.train_user_dict.items():
        if len(value)<=5:
            train_user_list.append(key)
    '''
    train_user_list = list(data.train_user_dict.keys())
    test_user_list = list(data.test_user_dict.keys())

    train_user_list = list(set(train_user_list) & set(test_user_list))
    if len(train_user_list) % data.cf_batch_size == 0:
        n_cf_batch = int(len(train_user_list) / data.cf_batch_size)
    else:
        n_cf_batch = len(train_user_list) // data.cf_batch_size + 1

    random.shuffle(train_user_list)
    for batch_num in range(n_cf_batch):
        time2 = time()
        batch_user_nodup = train_user_list[batch_num*data.cf_batch_size:(batch_num+1)*data.cf_batch_size]
        cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, batch_user_nodup)
        cf_batch_user_query, cf_batch_pos_item_query, cf_batch_neg_item_query = data.generate_cf_batch(data.test_user_dict, batch_user_nodup)
        batch_loss = []

        # update attention scores
        with torch.no_grad():
            att = model('calc_att', train_graph)
        train_graph.edata['att'] = att

        for i in range(len(cf_batch_user)):
            # print('task: ', i)
            cf_task_loss = model('meta_cf', train_graph, cf_batch_user[i], cf_batch_pos_item[i], cf_batch_neg_item[i],
                                 cf_batch_user_query[i], cf_batch_pos_item_query[i], cf_batch_neg_item_query[i])

            batch_loss.append(cf_task_loss)
            
        # update KGE
        kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, batch_user_nodup)
        if use_cuda:
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)
        kg_batch_loss = model('calc_kg_loss', kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

        loss = torch.stack(batch_loss).mean(0) + kg_batch_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cf_total_loss += loss.item()

        if (batch_num % args.cf_print_every) == 0:
            logging.info('Meta Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f}'.format(epoch, batch_num, n_cf_batch, time() - time2, loss.item()))
    logging.info('Meta Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

    # evaluate cf
    for epoch in range(500):
        time1 = time()
        
        with torch.no_grad():
            att = model('calc_att', train_graph)
        train_graph.edata['att'] = att

        # fine-turning
        cold_train_user_list = list(cold_data.train_user_dict.keys())
        if len(cold_train_user_list) % args.fine_tuning_batch_size == 0:
            ft_cf_batch = int(len(cold_train_user_list) / args.fine_tuning_batch_size)
        else:
            ft_cf_batch = len(cold_train_user_list) // args.fine_tuning_batch_size + 1
        random.shuffle(cold_train_user_list)

        for iter in range(ft_cf_batch):
            
            batch_user_nodup = cold_train_user_list[iter * args.fine_tuning_batch_size:(iter + 1) * args.fine_tuning_batch_size]

            # update CF
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = cold_data.generate_cf_batch(cold_data.train_user_dict, batch_user_nodup)
            if use_cuda:
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model('calc_cf_loss', train_graph, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, None, True)

            # update KGE
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = cold_data.generate_kg_batch(cold_data.train_kg_dict, batch_user_nodup)
            if use_cuda:
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)
            kg_batch_loss = model('calc_kg_loss', kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

            loss = cf_batch_loss + kg_batch_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time2 = time()
        _, precision10, recall10, ndcg10 = evaluate(model, train_graph, cold_data.train_user_dict, cold_data.test_user_dict, user_ids_batches, item_ids, args.K1)
        time3 = time()
        _, precision20, recall20, ndcg20 = evaluate(model, train_graph, cold_data.train_user_dict, cold_data.test_user_dict, user_ids_batches, item_ids, args.K)
        time4 = time()
        logging.info('Fine-tuning Evaluation: Epoch {:04d} | traing Time {:.1f}s | eval Time {:.1f}s | Precision10 {:.4f} Recall10 {:.4f} NDCG10 {:.4f} Precision20 {:.4f} Recall20 {:.4f} NDCG20 {:.4f}'.format(epoch, time2 - time1, time4 - time3, precision10, recall10, ndcg10, precision20, recall20, ndcg20))
        
        epoch_list.append(epoch)
        precision_list10.append(precision10)
        recall_list10.append(recall10)
        ndcg_list10.append(ndcg10)

        precision_list20.append(precision20)
        recall_list20.append(recall20)
        ndcg_list20.append(ndcg20)
        best_recall, should_stop = early_stopping(recall_list20, args.stopping_steps)

        if should_stop:
            break

        if recall_list20.index(best_recall) == len(recall_list20) - 1:
            save_model(model, args.save_dir, epoch, best_epoch)
            logging.info('Save model on epoch {:04d}!'.format(epoch))
            best_epoch = epoch


    # save model
    save_model(model, args.save_dir, epoch)

    # save metrics
    metrics = pd.DataFrame([epoch_list, precision_list10, recall_list10, ndcg_list10, precision_list20, recall_list20, ndcg_list20]).transpose()
    metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K1), 'recall@{}'.format(args.K1), 'ndcg@{}'.format(args.K1), 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K)]
    metrics.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

'''
def predict(args):
    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoader(args, logging)

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # load model
    model = MetaKG(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)
    # if n_gpu > 1:
    #     model = nn.parallel.DistributedDataParallel(model)

    # move graph data to GPU
    train_graph = data.train_graph
    train_nodes = torch.LongTensor(train_graph.ndata['id'])
    train_edges = torch.LongTensor(train_graph.edata['type'])
    if use_cuda:
        train_nodes = train_nodes.to(device)
        train_edges = train_edges.to(device)
    train_graph.ndata['id'] = train_nodes
    train_graph.edata['type'] = train_edges

    test_graph = data.test_graph
    test_nodes = torch.LongTensor(test_graph.ndata['id'])
    test_edges = torch.LongTensor(test_graph.edata['type'])
    if use_cuda:
        test_nodes = test_nodes.to(device)
        test_edges = test_edges.to(device)
    test_graph.ndata['id'] = test_nodes
    test_graph.edata['type'] = test_edges

    # predict
    cf_scores, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))
'''
if __name__ == '__main__':
    args = parse_Metakg_args()
    train(args)
    # predict(args)