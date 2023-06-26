from time import time
import dask_cudf
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import xgboost as xgb
import argparse

def process_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Command Line Arguments Parser')

    # Add the command line arguments
    parser.add_argument('--jit_unspill', default=None, type=bool, help='JIT unSpill: True or False')
    parser.add_argument('--device_memory_limit', default=None, type=str, help='Memory limit. e.g. 10GB')
    parser.add_argument('--rmm_pool_size', default=None, type=str, help='RMM memory pool size. e.g. 29GB')
    parser.add_argument('--protocol', default='tcp', choices=['tcp', 'ucx'], help='Protocol: tcp or tp')

    # Parse the command line arguments
    args = parser.parse_args()
    return args

def train_dask_xgboost(client):
    start = time()
    
    users = dask_cudf.read_parquet('/raid/otto/Otto-Comp/pqs/train_v152_*.pq').persist()#,split_row_groups=True)
    print(users.head())
    print('number of samples:',users.shape[0].compute())
    print(f'dataframe size: {users.memory_usage().sum().compute()/2**30:.1f} GB')


    VER = 186
    USE = 'clicks'
    FEATURES = users.columns[2:]
    TARS = [USE]
    FEATURES = [f for f in FEATURES if f not in TARS]
    print(len(FEATURES))
    print( FEATURES)
    print(TARS)

    dtrain = xgb.dask.DaskQuantileDMatrix(client, users[FEATURES], users['clicks'])

    FOLDS = 5
    SEED = 42

    LR = 0.1

    xgb_parms = { 
        'max_depth':4, 
        'learning_rate':LR, 
        'subsample':0.7,
        'colsample_bytree':0.5, 
        'eval_metric':'map',
        'objective':'binary:logistic',
        'scale_pos_weight':8,
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor',
        'random_state':SEED
    }

    output = xgb.dask.train(
            client,
            xgb_parms,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train")],
        )

    duration = time()-start
    print(f"All done! end-to-end time: {duration:.1f} seconds")

if __name__ == '__main__':
    args = process_arguments()
    print(args)
    cluster = LocalCUDACluster(device_memory_limit=args.device_memory_limit, 
                               jit_unspill=args.jit_unspill,
                               protocol=args.protocol, 
                               rmm_pool_size=args.rmm_pool_size)
    client = Client(cluster)
    train_dask_xgboost(client)