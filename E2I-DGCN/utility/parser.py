import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run E2I-DGCN.")
    parser.add_argument('--weights_path', nargs='?', default='model/',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='Movielens-100k',
                        help='Choose a dataset from {Gowalla, Yelp, Foursquare}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--alpha', nargs='?', default='[0.5, 0.5]',
                        help='weight of intent-aware subgraphs.')

    parser.add_argument('--beta', type=float, default=0.05,
                        help='coefficient of similarity term.')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')

    parser.add_argument('--model_type', nargs='?', default='E2I-DGCN',
                        help='Specify the name of model (E2I-DGCN).')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='E2I-DGCN',
                        help='Specify the type of the graph convolutional layer from {E2I-DGCN, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.2]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[5, 10, 15, 20, 25]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()
