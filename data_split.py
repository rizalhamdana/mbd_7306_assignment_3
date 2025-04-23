import pandas as pd

# Reload and sort the data
df = pd.read_csv("data/Groceries data train.csv")
df = df.dropna(how='all')
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df = df.sort_values('Date')
total_rows = len(df)

# Define the reusable splitting function
def split_dataset_by_date(df, ratio):
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

# Apply the function for 80-20 and 70-30 splits
train_70, dev_70, date_70 = split_dataset_by_date(df, 0.7)

# Save to CSV
train_70.to_csv("data/groceries_train_70.csv", index=False)
dev_70.to_csv("data/groceries_dev_30.csv", index=False)

