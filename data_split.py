from os import path
import pandas as pd

DATA_PATH = 'data'
DATA_FILE = 'Groceries data train.csv'

def split_dataset_by_date(df, ratio):
    '''
    Split dataset based on ratio and date
    '''
    df = df.sort_values("Date")
    total_rows = len(df)
    
    target_split = int(total_rows * ratio)
    cumulative_count = 0
    
    for date, count in df['Date'].value_counts().sort_index().items():
        cumulative_count += count
        
        if cumulative_count >= target_split:
            split_date = pd.to_datetime(date)
            
            train = df[df['Date'] < split_date]
            dev = df[df['Date'] >= split_date]
            
            return train, dev, split_date
    return df, pd.DataFrame(), None

df = pd.read_csv(path.join(DATA_PATH, DATA_FILE))

# Drop all empty rows
df = df.dropna(how='all')

# Convert date column to datetime for easier operations
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df = df.sort_values('Date')

# Apply the function for 70-30 splits
train_70, dev_70, date_70 = split_dataset_by_date(df, 0.7)

# Save to CSV
train_70.to_csv(path.join(DATA_PATH, 'groceries_train_70.csv'), index=False)
dev_70.to_csv(path.join(DATA_PATH, 'groceries_dev_30.csv'), index=False)
