import pandas as pd

def check_data_health(df: pd.DataFrame, name: str = "Data"):
    print(f"\n=== {name} Summary ===")
    print(f"Missing values:\n{df.isnull().sum()}\n")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}\n")
    return df

def handle_missing_and_duplicates(df: pd.DataFrame, name: str = "Data") -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.fillna(method='ffill').fillna(method='bfill')  
    print(f"\n=== {name} Cleaned ===")
    print(f"Remaining missing values: {df.isnull().sum().sum()}\n")
    return df

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    return df

def add_lag_features(df: pd.DataFrame, lags: list[int], group_col: str = "id") -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_col)["sales"].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: list[int], group_col: str = "id") -> pd.DataFrame:
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby(group_col)["sales"].shift(1).rolling(window).mean()
        )
    return df

def merge_external_data(df: pd.DataFrame, oil: pd.DataFrame,  
                        transactions: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(oil, on="date", how="left")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    df = df.merge(stores, on="store_nbr", how="left")  
    df = check_data_health(df, "Merged Data")
    df = handle_missing_and_duplicates(df, "Merged Data")
    return df

def add_holiday_feature(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    holidays_df = holidays_df[holidays_df['transferred'] == False]
    holidays_df = holidays_df[['date']].copy()
    holidays_df['is_holiday'] = 1

    df = df.merge(holidays_df, on='date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
    return df

def prepare_features(df: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame,
                     transactions: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    df = check_data_health(df, "Raw Data")
    df = handle_missing_and_duplicates(df, "Raw Data")
    df = create_date_features(df)
    df = add_holiday_feature(df, holidays)
    df = merge_external_data(df, oil, transactions, stores)
    df = add_lag_features(df, lags=[7, 14], group_col="id")
    df = add_rolling_features(df, windows=[7], group_col="id")
    df = check_data_health(df, "Final Feature Data")
    df = handle_missing_and_duplicates(df, "Final Feature Data")
    return df
