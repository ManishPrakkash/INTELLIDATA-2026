import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features: day_of_week, weekend_flag, month, week_no.
    festival_flag should be handled via join with external_factors, but we ensure it exists.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    df['day_of_week'] = df['date'].dt.dayofweek
    df['weekend_flag'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['week_no'] = df['date'].dt.isocalendar().week.astype(int)
    
    # festival_flag usually comes from external factors, initialize if missing
    if 'festival_flag' not in df.columns:
        df['festival_flag'] = 0
        
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag features based on corrected_demand: lag_1, lag_7, lag_14.
    """
    df = df.copy()
    
    # Sort just to be safe before shifting
    df = df.sort_values(by=['store_id', 'product_id', 'date'])
    
    group = df.groupby(['store_id', 'product_id'])['corrected_demand']
    df['lag_1'] = group.shift(1)
    df['lag_7'] = group.shift(7)
    df['lag_14'] = group.shift(14)
    
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling features based on corrected_demand: rolling_mean_7, rolling_mean_14, rolling_std_7.
    Crucially uses .shift(1) before .rolling() to prevent data leakage.
    """
    df = df.copy()
    df = df.sort_values(by=['store_id', 'product_id', 'date'])
    
    group = df.groupby(['store_id', 'product_id'])['corrected_demand']
    
    # .shift(1) prevents today's demand from leaking into today's rolling features
    shifted = group.shift(1)
    
    df['rolling_mean_7'] = shifted.rolling(window=7, min_periods=1).mean()
    df['rolling_mean_14'] = shifted.rolling(window=14, min_periods=1).mean()
    df['rolling_std_7'] = shifted.rolling(window=7, min_periods=1).std()
    
    return df

def handle_sparse_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For sparse history products (<14 days), fall back to category-level mean demand.
    Requires is_sparse_history flag.
    """
    df = df.copy()
    if 'is_sparse_history' not in df.columns:
        return df
        
    if 'category' in df.columns:
        # Calculate category mean from non-sparse items as a fallback
        category_mean = df[df['is_sparse_history'] == 0].groupby('category')['corrected_demand'].mean()
        
        # Where rolling/lag features are NaN AND is_sparse_history is True, fill with category mean
        # (This is a simplified approach, in practice you might want to fill specific columns)
        for col in ['lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_14']:
            if col in df.columns:
                df[col] = df.apply(
                    lambda row: category_mean.get(row['category'], 0) 
                    if pd.isna(row[col]) and row.get('is_sparse_history', False) 
                    else row[col],
                    axis=1
                )
    return df

def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds inventory features: days_of_inventory, inventory_to_demand_ratio, reorder_gap.
    """
    df = df.copy()
    
    # Guard divide by zero
    safe_rolling_mean = df['rolling_mean_7'].replace(0, np.nan)
    
    df['days_of_inventory'] = df['closing_stock'] / safe_rolling_mean
    # Cap infinity or fill NaN
    df['days_of_inventory'] = df['days_of_inventory'].fillna(0).clip(upper=999)
    
    df['inventory_to_demand_ratio'] = df['closing_stock'] / (df['corrected_demand'].replace(0, np.nan))
    df['inventory_to_demand_ratio'] = df['inventory_to_demand_ratio'].fillna(0).clip(upper=999)
    
    if 'reorder_lvl' in df.columns:
        df['reorder_gap'] = df['reorder_lvl'] - df['closing_stock']
        
    return df

def add_price_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds price/promo features: discount_pct, price_change.
    """
    df = df.copy()
    df = df.sort_values(by=['store_id', 'product_id', 'date'])
    
    group = df.groupby(['store_id', 'product_id'])['avg_selling_price']
    df['price_change'] = df['avg_selling_price'] - group.shift(1)
    
    if 'base_price' in df.columns:
        df['discount_pct'] = ((df['base_price'] - df['avg_selling_price']) / df['base_price']).clip(lower=0)
    else:
        df['discount_pct'] = 0.0 # Placeholder if base_price is not provided
        
    if 'promotion_flag' not in df.columns:
        df['promotion_flag'] = 0
        
    return df

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target: next_7_day_demand = forward sum of corrected_demand over the next 7 days.
    """
    df = df.copy()
    df = df.sort_values(by=['store_id', 'product_id', 'date'])
    
    # Calculate rolling sum looking forward
    # reverse the dataframe, apply rolling sum, then reverse back
    # Alternatively, use shift(-1) then rolling(7) and sum
    def forward_sum(x):
        return x.shift(-7).rolling(window=7, min_periods=1).sum()
        
    df['next_7_day_demand'] = df.groupby(['store_id', 'product_id'])['corrected_demand'].transform(
        lambda x: x.iloc[::-1].rolling(7, min_periods=1).sum().iloc[::-1].shift(-1)
    )
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline function to run all feature engineering steps.
    """
    # Ensure sorted order
    df = df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
    
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = handle_sparse_history(df)
    df = add_inventory_features(df)
    df = add_price_promo_features(df)
    
    # Target creation should only be used for training, but we generate it here
    df = create_target_variable(df)
    
    # Drop rows where target is NaN (the very end of the time series)
    # Be careful not to drop them if we need them for inference!
    # A robust pipeline would separate train and inference, but we keep it simple here.
    
    return df

if __name__ == "__main__":
    # Quick unit test / dry run on a dummy dataframe if needed
    print("Features pipeline loaded. Waiting for master_table.csv.")
