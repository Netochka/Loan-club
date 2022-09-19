import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Constants
data_filename = './data/accepted_2007_to_2018Q4.csv'
CLEARED_DATA_FILENAME = './data/accepted_2007_to_2018Q4_cleared.csv'
CHUNK_SIZE = 10 ** 5
IS_SAVE_TO_FILE = True  # True
CLEARED_COLUMNS_FILENAME = './data/cleared_columns.npy'
IS_EXTRACT_FEATURES = False  # True


def get_filtered_columns(input_data):
    filtered_data = input_data.dropna(how='all', axis=1)

    total_count = filtered_data.count().to_list()[0]
    max_empty_num = int(total_count * 0.5)  # maximum number of empty values in column
    empty_number_list = np.array(filtered_data.isna().sum())
    filtered_columns_indexes = np.where(empty_number_list < max_empty_num)[0]
    filtered_data = filtered_data.iloc[:, filtered_columns_indexes]  # Remove columns with sparse data
    filtered_columns = filtered_data.columns.to_list()

    return filtered_columns


def remove_missing_data(input_data):
    # removed missing columns
    # filtered_columns = np.load(CLEARED_COLUMNS_FILENAME)
    # clear_data = input_data[filtered_columns]

    # consider only business critical features
    # https://www.kaggle.com/code/mariiagusarova/feature-selection-techniques?scriptVersionId=104713173&cellId=11
    clear_data = input_data[['loan_amnt', 'term', 'int_rate', 'sub_grade', 'emp_length', 'annual_inc',
                             'loan_status', 'dti', 'mths_since_recent_inq', 'revol_util', 'bc_open_to_buy',
                             'bc_util', 'num_op_rev_tl']]

    clear_data = clear_data.dropna()
    return clear_data


def remove_outliers(input_data):
    q_low = input_data["annual_inc"].quantile(0.08)
    q_hi = input_data["annual_inc"].quantile(0.92)
    input_data = input_data[(input_data["annual_inc"] < q_hi) & (input_data["annual_inc"] > q_low)]
    input_data = input_data[(input_data['dti'] <= 45)]
    q_hi = input_data['bc_open_to_buy'].quantile(0.95)
    input_data = input_data[(input_data['bc_open_to_buy'] < q_hi)]
    input_data = input_data[(input_data['bc_util'] <= 160)]
    input_data = input_data[(input_data['revol_util'] <= 150)]
    input_data = input_data[(input_data['num_op_rev_tl'] <= 35)]
    return input_data


def categorical_processing(input_data):
    cleaner_app_type = {"term": {" 36 months": 1.0, " 60 months": 2.0},
                        "emp_length": {"< 1 year": 0.0, '1 year': 1.0, '2 years': 2.0, '3 years': 3.0, '4 years': 4.0,
                                       '5 years': 5.0, '6 years': 6.0, '7 years': 7.0, '8 years': 8.0, '9 years': 9.0,
                                       '10+ years': 10.0}
                        }

    return_data = input_data.replace(cleaner_app_type)

    return_data['sub_grade'] = return_data['sub_grade'].astype('category')
    return_data['sub_grade'] = return_data['sub_grade'].cat.codes

    return return_data


def process(input_data):
    clear_data = remove_missing_data(input_data)

    # Remove outliers
    clear_data = remove_outliers(clear_data)

    # Categorical features processing
    clear_data = categorical_processing(clear_data)

    # Add binary classification
    paid_loan_statuses = ['Current', 'Fully Paid']
    clear_data['loan_status'] = clear_data['loan_status'].apply(lambda x: int(x not in paid_loan_statuses))

    return clear_data


def get_columns_intersect(input):
    if len(input) < 2:
        return []
    result = set(input[0])
    for s in input[1:]:
        result.intersection_update(s)
    return np.array(list(result))


def get_extract_columns():
    columns_list = []
    with pd.read_csv(data_filename, chunksize=CHUNK_SIZE, low_memory=False) as reader:
        for chunk in reader:
            cleared_columns = get_filtered_columns(chunk)
            columns_list.append(cleared_columns)

    final_cleared_columns = get_columns_intersect(columns_list)

    np.save(CLEARED_COLUMNS_FILENAME, final_cleared_columns)


if __name__ == '__main__':
    if IS_EXTRACT_FEATURES:
        get_extract_columns()
    else:
        data_list = []
        with pd.read_csv(data_filename, chunksize=CHUNK_SIZE, low_memory=False) as reader:
            for chunk in reader:
                cleared_data = process(chunk)
                if not cleared_data.empty:
                    data_list.append(cleared_data)

        data = pd.concat(data_list)
        print(data.shape)


        # print(data['loan_status'].unique())
        # ['Charged Off' 'Current' 'Fully Paid' 'Late (31-120 days)'
        #  'Late (16-30 days)' 'In Grace Period' 'Default']

        # print(data.home_ownership.unique())  # ['MORTGAGE' 'RENT' 'OWN' 'ANY' 'NONE' 'OTHER']

        if IS_SAVE_TO_FILE:
            data.to_csv(CLEARED_DATA_FILENAME, index=False)
