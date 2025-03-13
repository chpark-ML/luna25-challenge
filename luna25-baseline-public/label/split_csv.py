import pandas as pd 
from sklearn.model_selection import train_test_split




if __name__ == '__main__':
    
    # binary: label
    # 0: 5608
    # 1: 555
    
    annot_df = pd.read_csv('/team/team_blu3/lung/data/2_public/LUNA25_Original/LUNA25_Public_Training_Development_Data.csv')
    
    # train, valid split by label class 
    train_df, valid_df = train_test_split(annot_df, test_size=0.2, stratify=annot_df['label'], random_state=2025, shuffle=True)
    
    save_dir = './'
    train_df.to_csv(save_dir + 'train.csv', index=False)
    valid_df.to_csv(save_dir + 'valid.csv', index=False)
    
    