import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MetaKG")

    # ===== dataset ===== #
    """
    Amazon_meta-train batch_size=4 meta_batch_size=2 num_inner_update=1 node_dropout=0.4
    Yelp2018_meta-train batch_size=5 meta_batch_size=2 num_inner_update=1 node_dropout=0.4
    LastFM_meta-train batch_size=6 meta_batch_size=2 num_inner_update=2 node_dropout=0.5
    """
    parser.add_argument("--dataset", nargs="?", default="last-fm",
                        help="Choose a dataset:[last-fm,amazon-book,alibaba,yelp2018,movie-lens,Book-Crossing]")
    parser.add_argument("--cold_scenario", nargs="?", default="user_item_cold", help="[user_cold, item_cold, user_item_cold, warm_up]")

    parser.add_argument("--data_path", nargs="?", default="datasets/", help="Input data path.")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--fine_tune_batch_size', type=int, default=512, help='fine tune batch size')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--meta_batch_size', type=int, default=2, help='meta batch size')
    parser.add_argument('--num_inner_update', type=int, default=2, help='number of inner update')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--meta_update_lr', type=float, default=0.001, help='meta update learning rate')
    parser.add_argument('--scheduler_lr', type=float, default=0.001, help='scheduler learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument('--use_gate', type=int, default=True, help='use gate or not')
    parser.add_argument('--use_pretrain', type=int, default=True, help='use pretrain data or not')
    parser.add_argument("--use_meta_model", type=bool, default=True, help="use trained meta model to adapt")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./model_para/", help="output directory for model")

    return parser.parse_args()
