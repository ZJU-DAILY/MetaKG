import math
import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable

from utility.parser_Metakg import parse_args
from utility.data_loader import load_data
from model.MetaKG import Recommender
from utility.evaluate import test
from utility.helper import early_stopping
from utility.scheduler import Scheduler
from collections import OrderedDict
from tqdm import tqdm

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
sample_num = 10

def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict

def get_feed_dict_meta(support_user_set):
    support_meta_set = []
    for key, val in support_user_set.items():
        feed_dict = []
        user = [int(key)] * sample_num
        if len(val) != sample_num:
            pos_item = np.random.choice(list(val), sample_num, replace=True)
        else:
            pos_item = val

        neg_item = []
        while True:
            tmp = np.random.randint(low=0, high=n_items, size=1)[0]
            if tmp not in val:
                neg_item.append(tmp)
            if len(neg_item) == sample_num:
                break
        feed_dict.append(np.array(user))
        feed_dict.append(np.array(list(pos_item)))
        feed_dict.append(np.array(neg_item))
        support_meta_set.append(feed_dict)

    return np.array(support_meta_set)  # [n_user, 3, 10]

def get_feed_kg(kg_graph):
    triplet_num = len(kg_graph)
    pos_hrt_id = np.random.randint(low=0, high=triplet_num, size=args.batch_size * sample_num)
    pos_hrt = kg_graph[pos_hrt_id]
    neg_t = np.random.randint(low=0, high=n_entities, size=args.batch_size*sample_num)

    return torch.LongTensor(pos_hrt[:,0]).to(device), torch.LongTensor(pos_hrt[:,1]).to(device),torch.LongTensor(pos_hrt[:,2]).to(device), torch.LongTensor(neg_t).to(device)

def convert_to_sparse_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape).to(device)

def get_net_parameter_dict(params):
    param_dict = dict()
    indexes = []
    for i, (name, param) in enumerate(params):
        if param.requires_grad:
            param_dict[name] = param.to(device)
            indexes.append(i)

    return param_dict, indexes

def update_moving_avg(mavg, reward, count):
    return mavg + (reward.item() - mavg) / (count + 1)

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    cold_scenario = args.cold_scenario  # the cold scenario adapted
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args, 'meta_training')
    adj_mat_list, mean_mat_list = mat_list
    cold_train_cf, cold_test_cf, cold_user_dict, cold_n_params, cold_graph, cold_mat_list = load_data(args, cold_scenario)
    cold_adj_mat_list, cold_mean_mat_list = cold_mat_list

    kg_graph = np.array(list(graph.edges))  # [-1, 3]
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    # train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    # test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    cold_train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in cold_train_cf], np.int32))
    # cold_test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in cold_test_cf], np.int32))
    
    """use pretrain data"""
    if args.use_pretrain:
        pre_path = args.data_path + 'pretrain/{}/mf.npz'.format(args.dataset)
        pre_data = np.load(pre_path)
        user_pre_embed = torch.tensor(pre_data['user_embed'])
        item_pre_embed = torch.tensor(pre_data['item_embed'])
    else:
        user_pre_embed = None
        item_pre_embed = None

    """init model"""
    model = Recommender(n_params, args, graph, user_pre_embed, item_pre_embed).to(device)
    names_weights_copy, indexes = get_net_parameter_dict(model.named_parameters())
    # print(names_weights_copy)
    scheduler = Scheduler(len(names_weights_copy), grad_indexes=indexes).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_update_lr)
    scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=args.scheduler_lr)

    """prepare feed data"""
    support_meta_set = get_feed_dict_meta(user_dict['train_user_set'])
    query_meta_set = get_feed_dict_meta(user_dict['test_user_set'])
    # shuffle
    index = np.arange(len(support_meta_set))
    np.random.shuffle(index)
    support_meta_set = support_meta_set[index]
    query_meta_set = query_meta_set[index]

    # support_cold_set = get_feed_dict_meta(cold_user_dict['train_user_set'])

    if args.use_meta_model:
        model.load_state_dict(torch.load('./model_para/meta_model_{}.ckpt'.format(args.dataset)))
    else:
        print("start meta training ...")
        """meta training"""
        # meta-training ui_interaction
        interact_mat = convert_to_sparse_tensor(mean_mat_list)
        model.interact_mat = interact_mat
        moving_avg_reward = 0

        model.train()
        iter_num = math.ceil(len(support_meta_set) / args.batch_size)
        train_s_t = time()
        for s in tqdm(range(iter_num)):
            batch_support = torch.LongTensor(support_meta_set[s * args.batch_size:(s + 1) * args.batch_size]).to(device)
            batch_query = torch.LongTensor(query_meta_set[s * args.batch_size:(s + 1) * args.batch_size]).to(device)

            pt = int(s / iter_num * 100)
            if len(batch_support) > args.meta_batch_size:
                task_losses, weight_meta_batch = scheduler.get_weight(batch_support, batch_query, model, pt)
                torch.cuda.empty_cache()
                task_prob = torch.softmax(weight_meta_batch.reshape(-1), dim=-1)
                selected_tasks_idx = scheduler.sample_task(task_prob, args.meta_batch_size)
                batch_support = batch_support[selected_tasks_idx]
                batch_query = batch_query[selected_tasks_idx]

            selected_losses = scheduler.compute_loss(batch_support, batch_query, model)
            meta_batch_loss = torch.mean(selected_losses)

            """KG loss"""
            h, r, pos_t, neg_t = get_feed_kg(kg_graph)
            kg_loss = model.forward_kg(h, r, pos_t, neg_t)
            batch_loss = kg_loss + meta_batch_loss

            """update scheduler"""
            loss_scheduler = 0
            for idx in selected_tasks_idx:
                loss_scheduler += scheduler.m.log_prob(idx.cuda())
            reward = meta_batch_loss
            loss_scheduler *= (reward - moving_avg_reward)
            moving_avg_reward = update_moving_avg(moving_avg_reward, reward, s)

            scheduler_optimizer.zero_grad()
            loss_scheduler.backward(retain_graph=True)
            scheduler_optimizer.step()

            """update network"""
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if args.save:
            torch.save(model.state_dict(), args.out_dir + 'meta_model_' + args.dataset + '.ckpt')

        train_e_t = time()
        print('meta_training_time: ', train_e_t-train_s_t)
    

    """fine tune"""
    # adaption ui_interaction
    cold_interact_mat = convert_to_sparse_tensor(cold_mean_mat_list)
    model.interact_mat = cold_interact_mat

    # reset lr
    for g in optimizer.param_groups:
        g['lr'] = args.lr

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    print("start fine tune...")
    for epoch in range(args.epoch):
        # shuffle training data
        index = np.arange(len(cold_train_cf))
        np.random.shuffle(index)
        cold_train_cf_pairs = cold_train_cf_pairs[index]

        model.train()
        loss = 0
        iter_num = math.ceil(len(cold_train_cf) / args.fine_tune_batch_size)
        train_s_t = time()
        for s in tqdm(range(iter_num)):
            batch = get_feed_dict(cold_train_cf_pairs,
                                  s*args.fine_tune_batch_size, (s+1) * args.fine_tune_batch_size,
                                  cold_user_dict['train_user_set'])
            batch_loss = model(batch, is_apapt=True)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        train_e_t = time()

        if epoch % 5 == 0 or epoch == 1:
            """testing"""
            model.eval()
            torch.cuda.empty_cache()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, cold_user_dict, cold_n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'],])
            print(train_res)

            f = open('./result/{}_{}_bt{}_lr{}.txt'.format(args.dataset, cold_scenario, args.fine_tune_batch_size, args.lr), 'a+')
            f.write(str(train_res) + '\n')
            f.close()

            # early stopping.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=20)
            if should_stop:
                break

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
