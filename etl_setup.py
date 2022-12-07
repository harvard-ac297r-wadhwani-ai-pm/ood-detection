import os
import argparse
import logging
import glob
import pandas as pd
from data_utils.data_setup import generate_val_df, creates_dirs, generate_splits, reorganize_train_test_splits, pipeline_copy_resize_imgs, generate_train_test

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser('Resize, reorganize, and generate train-test splits')
    parser.add_argument('--src_repo', type=str, default='data', help='src data folder with opendata images [%(default)s]')
    return parser.parse_args()

def main(args):
    SRC_DIR = f'{args.src_repo}' + '/data'
    logging.basicConfig(filename='output.log', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    
    logging.info('Creates directories')
    DEST_DIR = BASE_FOLDER + '/data'
    creates_dirs(BASE_FOLDER)
    
    logging.info('Load files')
    img_paths = glob.glob(SRC_DIR + '/images/*/*.jpg')
    ref_df = pd.read_csv(BASE_FOLDER + '/data_utils/id_ec_ood_split.csv')

    logging.info('Matches files between directory and reference directory')
    validate_df = generate_val_df(img_paths,ref_df)
 
    logging.info('Generates the split')
    id_subset, ec_subset, ood_subset = generate_splits(validate_df)

    logging.info('Pipeline to resize and copy images')
    DEST_DIR = BASE_FOLDER + '/data'
    pipeline_copy_resize_imgs(DEST_DIR,id_subset,ec_subset,ood_subset)

    logging.info('Generates bollworms-clean-*')
    X_clean_train, X_clean_test = generate_train_test(DEST_DIR,clean_flag=True)
    reorganize_train_test_splits(DEST_DIR,X_clean_train,'bollworms-clean-train','X_clean_train.csv')
    reorganize_train_test_splits(DEST_DIR,X_clean_test,'bollworms-clean-test','X_clean_test.csv')

    logging.info('Generates bollworms-*')
    X_train, X_test = generate_train_test(DEST_DIR,clean_flag=False)
    reorganize_train_test_splits(DEST_DIR,X_train,'bollworms-train','X_train.csv')
    reorganize_train_test_splits(DEST_DIR,X_test,'bollworms-test','X_test.csv')

if __name__ == '__main__':
    args = parse_args()
    print(f'Fetching files from opendata (src repo): {args.src_repo}')
    main(args)
    print('Finished generating organized set of images in /data')