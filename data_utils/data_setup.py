import os
import glob
import shutil
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from functools import partial
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

def generate_val_df(img_paths,ref_df):
    '''
    Validates match between open data images images and dataframe that contains sorted images via Harvard IACS Capstone-Wadhwani partnership  
    '''
    src_df = pd.DataFrame()
    src_df['img_path'] = img_paths
    src_df['img_filename'] = src_df['img_path'].apply(lambda x: x.split('/')[-1])
    validate_df = ref_df.merge(src_df,on=['img_filename'],how='left')
    validate_df.drop(columns=['Unnamed: 0'],inplace=True)
    assert validate_df.shape[0] == len(set(validate_df['img_filename'])) #confirms no duplicate entries 
    return validate_df

def creates_dirs(BASE_FOLDER):
    '''
    Generates ID/EC/OOD folders in the DATA_DIR directory 
    '''
    dir_list = ['data/reorg_imgs/ID', 'data/reorg_imgs/EC', 'data/reorg_imgs/OOD','data/bollworms-clean-train/ID','data/bollworms-clean-train/OOD','data/bollworms-clean-test/ID','data/bollworms-clean-test/OOD',
    'data/bollworms-train/ID','data/bollworms-train/OOD','data/bollworms-test/ID','data/bollworms-test/OOD']
    for dir_i in dir_list:
        os.makedirs(BASE_FOLDER + '/'+ dir_i,exist_ok=True)
    
def copy_imgs(DATA_DIR, img_paths, dest_dir_name):
    '''
    Loops through all images to resize and save image 
    '''
    dst_dir = DATA_DIR + '/reorg_imgs/' + dest_dir_name
    for img in tqdm(img_paths):
        resize_img(img,dst_dir)

def resize_img(source_path,new_path,resize_dim=256):
    '''
    Resize function to resize iamges and save to new directory
    Default dimension: 256x256
    Assumes square resize
    '''
    img_name = source_path.split('/')[-1]
    image_open = open(source_path, 'rb')
    read_image = image_open.read()
    image_decode = tf.image.decode_jpeg(read_image)
    resize_image = tf.image.resize(image_decode, [resize_dim, resize_dim], method='nearest')
    save_new_dir = new_path + '/' + img_name
    tf.keras.preprocessing.image.save_img(save_new_dir,resize_image)

def generate_splits(validate_df):
    '''
    Splits up data in ID, EC, and OOD for resizing and copy pipeline  
    '''
    id_split = validate_df[validate_df['label'] == 'id']
    ec_split = validate_df[validate_df['label'] == 'ec']
    ood_split = validate_df[validate_df['label'] == 'ood']
    return id_split, ec_split, ood_split

def pipeline_copy_resize_imgs(DATA_DIR, id, ec, ood):
    ''' 
    Generates pipeline to resize & copy images into corresponding folders
    '''
    print("Resize & sort ID images")
    copy_imgs(DATA_DIR,id['img_path'],'ID')
    print("Resize & sort EC images")
    copy_imgs(DATA_DIR,ec['img_path'],'EC')
    print("Resize & sort OOD images")
    copy_imgs(DATA_DIR,ood['img_path'],'OOD')


def generate_train_test(DATA_DIR,clean_flag=False):
    '''
    Generates train test splits
    clean_flag = True: Define EC as OOD
    clean_flag = False: Define EC as ID
    Motivation:  
        bollworms-clean only contains “clean” images of bollworms as ID (define EC as OOD),
        bollworms-test can contain other edge cases as part of ID
    '''
    id_dir = DATA_DIR + '/reorg_imgs/ID/*.jpg'
    ec_dir = DATA_DIR + '/reorg_imgs/EC/*.jpg'
    ood_dir = DATA_DIR + '/reorg_imgs/OOD/*.jpg'

    # extract img paths
    id_paths = glob.glob(id_dir)
    ec_paths = glob.glob(ec_dir)
    ood_paths = glob.glob(ood_dir)

    id = pd.DataFrame()
    id['img_path'] = id_paths 
    id['label'] = 1

    ood = pd.DataFrame()
    ood['img_path'] = ood_paths
    ood['label'] = 0

    ec = pd.DataFrame()
    ec['img_path'] = ec_paths

    if clean_flag == True:
        # if clean, then consider EC as OOD to ensure ID set is clean
        print("Define EC as OOD")
        ec['label'] = 0
    else:
        # if not clean, then consider EC as ID, introducing variation into ID set 
        print("Define EC as ID")
        ec['label'] = 1

    df = pd.concat((id,ood,ec))
    X = df
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, stratify=X['label'],random_state=42)

    return X_train, X_test



def reorganize_train_test_splits(DATA_DIR, df, reorg_dir, csv_name):
    dest_dir = DATA_DIR + '/' + reorg_dir
    dest_dir_id = dest_dir + '/ID'
    dest_dir_ood = dest_dir + '/OOD'

    train_id = df[df['label'] == 1]
    train_ood = df[df['label'] == 0]
    csv_path = dest_dir+'/'+ csv_name + '.csv'
    df.to_csv(csv_path)

    print("Write to:",dest_dir)
    print(dest_dir_id)
    print(dest_dir_ood)
    
    for img_file in tqdm(train_ood['img_path']):
        shutil.copy(img_file, dest_dir_ood)
    for img_file in tqdm(train_id['img_path']):
        shutil.copy(img_file, dest_dir_id)