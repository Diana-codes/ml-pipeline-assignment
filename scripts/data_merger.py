import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import re

def load_and_merge_datasets():
    """
    Load and merge the provided customer social profiles and transactions datasets
    """
    print("Loading provided datasets...")
    
    # Load customer social profiles
    try:
        # If you have the data as CSV files
        social_profiles = pd.read_csv('data/customer_social_profiles.csv')
        transactions = pd.read_csv('data/customer_transactions.csv')
        print("✓ Loaded datasets from CSV files")
    except FileNotFoundError:
        print("Creating datasets from provided data...")
        
        # Customer Social Profiles Data (from your assignment)
        social_data = {
            'customer_id': ['A178', 'A190', 'A150', 'A162', 'A197', 'A151', 'A137', 'A196', 'A187', 'A178',
                           'A129', 'A150', 'A180', 'A104', 'A128', 'A103', 'A109', 'A155', 'A116', 'A173',
                           'A116', 'A183', 'A187', 'A168', 'A133', 'A105', 'A152', 'A165', 'A176', 'A142',
                           'A174', 'A122', 'A154', 'A179', 'A194', 'A174', 'A115', 'A107', 'A103', 'A103',
                           'A155', 'A124', 'A166', 'A195', 'A166', 'A126', 'A192', 'A131', 'A149', 'A160',
                           'A150', 'A118', 'A120', 'A104', 'A181', 'A191', 'A141', 'A160', 'A121', 'A120',
                           'A169', 'A100', 'A104', 'A111', 'A189', 'A145', 'A133', 'A148', 'A177', 'A189',
                           'A144', 'A126', 'A172', 'A125', 'A146', 'A185', 'A155', 'A193', 'A162', 'A147',
                           'A160', 'A180', 'A125', 'A135', 'A100', 'A107', 'A198', 'A151', 'A178', 'A146',
                           'A155', 'A185', 'A113', 'A189', 'A127', 'A186', 'A177', 'A187', 'A101', 'A125'],
            'social_media_platform': ['LinkedIn', 'Twitter', 'Facebook', 'Twitter', 'Twitter', 'TikTok', 'LinkedIn', 'Instagram', 'LinkedIn', 'LinkedIn',
                                     'Twitter', 'LinkedIn', 'Facebook', 'Twitter', 'Facebook', 'Instagram', 'Twitter', 'Instagram', 'Twitter', 'TikTok',
                                     'LinkedIn', 'TikTok', 'Instagram', 'LinkedIn', 'Twitter', 'LinkedIn', 'Facebook', 'LinkedIn', 'Facebook', 'LinkedIn',
                                     'Facebook', 'Instagram', 'TikTok', 'Twitter', 'LinkedIn', 'TikTok', 'Twitter', 'Twitter', 'Facebook', 'Instagram',
                                     'Instagram', 'TikTok', 'Instagram', 'LinkedIn', 'Instagram', 'TikTok', 'Instagram', 'LinkedIn', 'Facebook', 'TikTok',
                                     'Facebook', 'Facebook', 'Facebook', 'Twitter', 'Twitter', 'Facebook', 'TikTok', 'LinkedIn', 'LinkedIn', 'Instagram',
                                     'TikTok', 'Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'Twitter', 'Twitter', 'Facebook', 'LinkedIn', 'TikTok',
                                     'Twitter', 'Facebook', 'TikTok', 'LinkedIn', 'Twitter', 'TikTok', 'TikTok', 'TikTok', 'Twitter', 'Twitter',
                                     'Instagram', 'LinkedIn', 'Facebook', 'TikTok', 'Instagram', 'Facebook', 'Instagram', 'Twitter', 'TikTok', 'Facebook',
                                     'Facebook', 'Facebook', 'Facebook', 'Facebook', 'Instagram', 'TikTok', 'Twitter', 'Twitter', 'Twitter', 'Twitter'],
            'engagement_score': [74, 82, 96, 89, 92, 61, 93, 85, 97, 53, 54, 86, 57, 91, 98, 77, 80, 58, 78, 63,
                                89, 90, 71, 60, 72, 50, 95, 86, 70, 75, 95, 85, 72, 50, 89, 64, 70, 96, 58, 58,
                                59, 75, 90, 84, 74, 75, 60, 87, 51, 56, 67, 76, 83, 76, 66, 92, 93, 73, 74, 56,
                                55, 73, 82, 98, 78, 92, 71, 75, 77, 99, 70, 98, 56, 66, 69, 90, 98, 69, 71, 77,
                                89, 98, 56, 50, 81, 62, 79, 72, 68, 81, 79, 78, 98, 94, 78, 79, 65, 89, 68, 67],
            'purchase_interest_score': [4.9, 4.8, 1.6, 2.6, 2.3, 1.3, 3.5, 3.9, 4.4, 1.5, 4.5, 3.6, 3.8, 4.6, 3.5, 2.3, 4.3, 2.5, 1.1, 4.3,
                                       2.4, 4.1, 2.5, 4.4, 1.9, 4.9, 4.1, 1.5, 3.3, 4.9, 2.9, 1.7, 2.9, 3.0, 4.0, 3.8, 2.6, 1.9, 3.6, 2.7,
                                       1.5, 4.6, 3.8, 3.7, 4.3, 3.1, 4.3, 3.0, 1.3, 2.6, 3.0, 3.9, 1.4, 1.6, 2.1, 2.1, 3.9, 2.0, 3.5, 3.0,
                                       3.3, 4.4, 2.6, 4.1, 2.7, 2.4, 4.8, 1.7, 1.4, 4.1, 1.3, 4.3, 2.7, 4.7, 3.9, 3.4, 4.8, 2.6, 4.8, 1.5,
                                       2.9, 2.1, 3.2, 1.6, 4.4, 4.4, 4.8, 1.6, 3.5, 2.9, 2.4, 2.4, 2.7, 3.8, 3.4, 2.8, 1.3, 1.3, 1.0, 4.9],
            'review_sentiment': ['Positive', 'Neutral', 'Positive', 'Positive', 'Neutral', 'Neutral', 'Neutral', 'Negative', 'Negative', 'Neutral',
                                'Neutral', 'Neutral', 'Negative', 'Negative', 'Positive', 'Positive', 'Neutral', 'Negative', 'Negative', 'Negative',
                                'Neutral', 'Positive', 'Neutral', 'Positive', 'Positive', 'Neutral', 'Positive', 'Positive', 'Negative', 'Positive',
                                'Neutral', 'Neutral', 'Positive', 'Negative', 'Negative', 'Positive', 'Positive', 'Neutral', 'Positive', 'Negative',
                                'Positive', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Positive', 'Positive', 'Positive', 'Positive',
                                'Neutral', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Neutral', 'Positive', 'Positive',
                                'Positive', 'Negative', 'Neutral', 'Negative', 'Negative', 'Neutral', 'Negative', 'Neutral', 'Positive', 'Neutral',
                                'Positive', 'Negative', 'Negative', 'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Positive', 'Neutral',
                                'Neutral', 'Negative', 'Neutral', 'Positive', 'Neutral', 'Positive', 'Positive', 'Neutral', 'Negative', 'Neutral',
                                'Positive', 'Positive', 'Neutral', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Neutral', 'Positive']
        }
        
        # Customer Transactions Data (sample from your assignment)
        transaction_data = {
            'customer_id_legacy': [151, 192, 114, 171, 160, 120, 182, 186, 174, 174, 187, 199, 123, 202, 121, 152, 101, 187, 129, 137,
                                  101, 163, 159, 120, 132, 175, 157, 121, 188, 148, 190, 158, 141, 191, 159, 179, 114, 161, 161, 146,
                                  161, 150, 154, 163, 102, 150, 106, 120, 172, 138, 117, 103, 188, 159, 113, 108, 189, 152, 101, 183,
                                  191, 159, 170, 143, 107, 146, 134, 177, 180, 135, 149, 103, 101, 105, 153, 103, 153, 192, 162, 117,
                                  189, 143, 133, 173, 161, 199, 113, 194, 147, 114, 171, 177, 186, 161, 139, 184, 179, 181, 152, 123],
            'transaction_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
                              1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040,
                              1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060,
                              1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080,
                              1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100],
            'purchase_amount': [408, 332, 442, 256, 564, 395, 791, 429, 228, 112, 401, 280, 290, 101, 145, 437, 271, 456, 280, 286,
                               192, 220, 78, 85, 62, 209, 376, 236, 292, 135, 333, 115, 219, 494, 111, 490, 183, 333, 77, 157,
                               193, 389, 335, 495, 380, 177, 397, 280, 239, 274, 434, 426, 332, 495, 170, 165, 282, 308, 408, 247,
                               460, 186, 367, 214, 274, 356, 283, 221, 201, 364, 423, 209, 145, 282, 229, 162, 367, 491, 101, 317,
                               344, 435, 436, 162, 150, 162, 489, 130, 236, 162, 151, 179, 269, 103, 392, 273, 274, 434, 452, 175],
            'purchase_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10',
                             '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20',
                             '2024-01-21', '2024-01-22', '2024-01-23', '2024-01-24', '2024-01-25', '2024-01-26', '2024-01-27', '2024-01-28', '2024-01-29', '2024-01-30',
                             '2024-01-31', '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04', '2024-02-05', '2024-02-06', '2024-02-07', '2024-02-08', '2024-02-09',
                             '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16', '2024-02-17', '2024-02-18', '2024-02-19',
                             '2024-02-20', '2024-02-21', '2024-02-22', '2024-02-23', '2024-02-24', '2024-02-25', '2024-02-26', '2024-02-27', '2024-02-28', '2024-02-29',
                             '2024-03-01', '2024-03-02', '2024-03-03', '2024-03-04', '2024-03-05', '2024-03-06', '2024-03-07', '2024-03-08', '2024-03-09', '2024-03-10',
                             '2024-03-11', '2024-03-12', '2024-03-13', '2024-03-14', '2024-03-15', '2024-03-16', '2024-03-17', '2024-03-18', '2024-03-19', '2024-03-20',
                             '2024-03-21', '2024-03-22', '2024-03-23', '2024-03-24', '2024-03-25', '2024-03-26', '2024-03-27', '2024-03-28', '2024-03-29', '2024-03-30',
                             '2024-03-31', '2024-04-01', '2024-04-02', '2024-04-03', '2024-04-04', '2024-04-05', '2024-04-06', '2024-04-07', '2024-04-08', '2024-04-09'],
            'product_category': ['Sports', 'Electronics', 'Electronics', 'Clothing', 'Clothing', 'Groceries', 'Sports', 'Clothing', 'Electronics', 'Sports',
                                'Sports', 'Electronics', 'Clothing', 'Electronics', 'Sports', 'Books', 'Books', 'Groceries', 'Electronics', 'Electronics',
                                'Groceries', 'Groceries', 'Groceries', 'Sports', 'Electronics', 'Sports', 'Groceries', 'Electronics', 'Sports', 'Sports',
                                'Groceries', 'Electronics', 'Groceries', 'Electronics', 'Books', 'Clothing', 'Clothing', 'Clothing', 'Groceries', 'Books',
                                'Electronics', 'Sports', 'Electronics', 'Sports', 'Electronics', 'Books', 'Sports', 'Groceries', 'Electronics', 'Electronics',
                                'Sports', 'Groceries', 'Groceries', 'Books', 'Groceries', 'Groceries', 'Groceries', 'Clothing', 'Books', 'Electronics',
                                'Sports', 'Electronics', 'Books', 'Sports', 'Books', 'Groceries', 'Sports', 'Groceries', 'Electronics', 'Electronics',
                                'Sports', 'Sports', 'Books', 'Books', 'Groceries', 'Sports', 'Electronics', 'Books', 'Books', 'Electronics',
                                'Books', 'Groceries', 'Sports', 'Electronics', 'Sports', 'Books', 'Books', 'Electronics', 'Groceries', 'Clothing',
                                'Electronics', 'Clothing', 'Clothing', 'Groceries', 'Clothing', 'Clothing', 'Groceries', 'Clothing', 'Clothing', 'Clothing'],
            'customer_rating': [2.3, 4.2, 2.1, 2.8, 1.3, 1.1, 4.9, 4.3, 3.8, 2.6, 1.7, 1.6, 2.0, 3.2, 3.9, 3.6, 2.1, 4.8, 4.0, 3.2,
                               3.4, 2.7, 2.0, 2.4, 4.0, 1.1, 1.5, 1.2, 1.2, 4.4, 3.8, 2.9, 1.4, 3.0, 2.9, 1.7, 2.7, 2.6, 1.0, 3.5,
                               1.2, 1.0, 1.6, 3.0, 4.4, 3.6, 1.7, 1.3, 3.6, 1.1, 3.3, 4.8, 3.3, 2.6, 3.6, 2.8, 3.2, 4.8, 2.5, 4.8,
                               4.6, 1.8, 1.3, 1.4, 1.1, 1.4, 3.7, 1.3, 2.3, 1.0, 1.1, 4.3, 2.1, 1.5, 3.8, 3.5, 4.5, 3.9, 4.2, 2.1,
                               1.0, 4.0, 4.2, 5.0, 2.7, 2.7, 4.9, 2.4, 1.0, 4.4, 2.7, 4.0, 4.0, 1.3, 4.6, 3.0, 4.8, 1.0, 4.0, 2.7]
        }
        
        # Create DataFrames
        social_profiles = pd.DataFrame(social_data)
        transactions = pd.DataFrame(transaction_data)
        
        # Save the datasets
        os.makedirs('data', exist_ok=True)
        social_profiles.to_csv('data/customer_social_profiles.csv', index=False)
        transactions.to_csv('data/customer_transactions.csv', index=False)
        print("✓ Created and saved datasets from provided data")
    
    return social_profiles, transactions

def feature_engineering(social_profiles, transactions):
    """
    Engineer features from merged dataset using the real provided data
    """
    print("Engineering features from real datasets...")
    
    # Convert purchase_date to datetime
    transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
    
    # Create a mapping between customer_id formats (A178 -> 178)
    # Extract numeric part from customer_id in social profiles - Fixed regex
    social_profiles['customer_id_numeric'] = social_profiles['customer_id'].str.extract(r'(\d+)').astype(int)
    
    # Aggregate transaction data per customer
    transaction_agg = transactions.groupby('customer_id_legacy').agg({
        'purchase_amount': ['mean', 'sum', 'std', 'count'],
        'customer_rating': ['mean', 'std'],
        'transaction_id': 'count'
    }).round(2)
    
    # Flatten column names
    transaction_agg.columns = ['_'.join(col).strip() for col in transaction_agg.columns]
    transaction_agg = transaction_agg.reset_index()
    transaction_agg.rename(columns={'customer_id_legacy': 'customer_id_numeric'}, inplace=True)
    
    # Get most frequent product category per customer
    most_frequent_category = transactions.groupby('customer_id_legacy')['product_category'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    most_frequent_category.columns = ['customer_id_numeric', 'preferred_category']
    
    # Calculate days since last purchase
    max_date = transactions['purchase_date'].max()
    last_purchase = transactions.groupby('customer_id_legacy')['purchase_date'].max().reset_index()
    last_purchase['days_since_last_purchase'] = (max_date - last_purchase['purchase_date']).dt.days
    last_purchase.rename(columns={'customer_id_legacy': 'customer_id_numeric'}, inplace=True)
    
    # Merge all data
    merged_data = social_profiles.merge(transaction_agg, on='customer_id_numeric', how='left')
    merged_data = merged_data.merge(most_frequent_category, on='customer_id_numeric', how='left')
    merged_data = merged_data.merge(last_purchase[['customer_id_numeric', 'days_since_last_purchase']], 
                                   on='customer_id_numeric', how='left')
    
    # Fill NaN values for customers without transactions
    merged_data = merged_data.fillna(0)
    
    # Create additional engineered features
    merged_data['engagement_per_purchase'] = merged_data['engagement_score'] / (merged_data['purchase_amount_mean'] + 1)
    merged_data['interest_rating_ratio'] = merged_data['purchase_interest_score'] / (merged_data['customer_rating_mean'] + 1)
    merged_data['purchase_frequency'] = merged_data['transaction_id_count'] / (merged_data['days_since_last_purchase'] + 1)
    
    return merged_data

def preprocess_for_ml(merged_data):
    """
    Preprocess data for machine learning - Fixed data type issues
    """
    print("Preprocessing for ML...")
    
    # Encode categorical variables
    le_platform = LabelEncoder()
    le_sentiment = LabelEncoder()
    le_category = LabelEncoder()
    
    merged_data['platform_encoded'] = le_platform.fit_transform(merged_data['social_media_platform'])
    merged_data['sentiment_encoded'] = le_sentiment.fit_transform(merged_data['review_sentiment'])
    
    # Handle preferred_category - convert all to strings and handle NaN/0 values
    merged_data['preferred_category'] = merged_data['preferred_category'].astype(str)
    merged_data['preferred_category'] = merged_data['preferred_category'].replace(['0', '0.0', 'nan'], 'Unknown')
    
    # Now encode the cleaned preferred_category
    merged_data['preferred_category_encoded'] = le_category.fit_transform(merged_data['preferred_category'])
    
    # Select features for modeling
    feature_columns = [
        'engagement_score', 'purchase_interest_score', 'platform_encoded', 'sentiment_encoded',
        'purchase_amount_mean', 'purchase_amount_sum', 'purchase_amount_std', 'purchase_amount_count',
        'customer_rating_mean', 'customer_rating_std', 'transaction_id_count',
        'days_since_last_purchase', 'engagement_per_purchase', 'interest_rating_ratio', 'purchase_frequency'
    ]
    
    # Remove any rows with NaN values in feature columns
    merged_data_clean = merged_data.dropna(subset=feature_columns)
    
    # Replace any remaining inf values with 0
    merged_data_clean = merged_data_clean.replace([np.inf, -np.inf], 0)
    
    X = merged_data_clean[feature_columns]
    y = merged_data_clean['preferred_category_encoded']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, merged_data_clean, le_category, scaler

def main():
    """
    Main function to execute data merging and preprocessing
    """
    print("Loading and processing real assignment datasets...")
    social_profiles, transactions = load_and_merge_datasets()
    
    print("Engineering features...")
    merged_data = feature_engineering(social_profiles, transactions)
    
    print("Preprocessing for ML...")
    X, y, processed_data, label_encoder, scaler = preprocess_for_ml(merged_data)
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    processed_data.to_csv('data/merged_dataset.csv', index=False)
    
    # Save train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    print(f"Data preprocessing complete!")
    print(f"Dataset shape: {processed_data.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Unique customers: {len(processed_data['customer_id'].unique())}")
    print(f"Product categories: {processed_data['preferred_category'].value_counts()}")

if __name__ == "__main__":
    main()
