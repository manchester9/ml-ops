import os
import logging
from churn_library import *
import churn_library as cls
from sklearn.model_selection import train_test_split
import pandas as pd

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("SUCCESS: Testing import_data")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
		logging.info('SUCCESS: There are rows and columns')
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	# set expectations for output files
	file_names = [
        './images/eda/Churn_histogram.png',
        './images/eda/age_histogram.png',
        './images/eda/marital_status_bins.png',
        './images/eda/total_trans_ct_histogram_density_plot.png',
        './images/eda/variable_heatmap.png'
    ]
	try:
		for file in file_names:
			assert os.path.exists(file)
		logging.info('SUCCESS: All file names are present in directory')
	except AssertionError as err:
		logging.error(f'File not found: {file}')		
		raise err
	
		
	# 	# check if expected files exist
	# 	assert all(os.path.isfile(file_name) for file_name in file_names)

    #     # check if expected files are not empty
    #     assert all(os.stat(file_name).st_size >
    #                0 for file_name in file_names)

    #     logging.info("Testing test_eda: SUCCESS")
    # except AssertionError as err:
    #     logging.error(
    #         "Testing test_eda: FAILED Some files were not found or empty %s",
    #         list(
    #             filter(
    #                 lambda x: not os.path.isfile(x),
    #                 file_names)))
    #     raise err


def test_encoder_helper(encoder_helper, df):
	
	col_names = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]
	category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]

	df = encoder_helper(df, category_list, 'Churn')
	
	try:
		for col in col_names:
			assert col in df.columns
		logging.info('SUCCESS: Columns were added to the end with _churn ')
	except AssertionError as err:
		logging.error(f'Column not found: {col}')		
		raise err


def test_perform_feature_engineering(perform_feature_engineering, df, KEEP_COLS):
	
	X_train, X_test, y_train, y_test, X, y = perform_feature_engineering(
            df, KEEP_COLS)

	y_train.columns = ['Churn']
	y_test.columns = ['Churn']
	lst = [X_train, X_test, y_train, y_test]
		
	try:
		for i in range(len(lst)):
			if i <= 1:
				assert lst[i].shape[0] > 0
				assert lst[i].shape[1] > 0
			else:
				assert lst[i].shape[0] > 0
				#assert lst[i].shape[1] == 1
		logging.info('SUCCESS: Dataframes are not empty')

	except AssertionError as err:
		logging.error(f'Testing ingestion file: FAILED The {lst[i]} file doesnt appear to have rows and columns')
		raise err
	except IndexError as ind:
		# for element in lst:
		logging.error(f'Variable element: FAILED element {i} with shape {lst[i].shape} doesnt have the right index')
		print(lst[i].head(4))


def test_train_models(train_models,X,y):
	### needs X and y from perform_feature_engineering for train_test_split function
	## the X and y being provided are inadequate somehow
	try:
		# train_models()
		logging.info("SUCCESS: Testing train_models")
	#	y = df['Churn']  ## comment out?
	#	X = pd.DataFrame()  ## comment out?
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
		train_models(X_train, X_test, y_train, y_test, X, y, cls.params)
	except MemoryError as err:
		logging.error(
            "Testing train_models: Out of memory while train the models")
		raise err


if __name__ == "__main__":
	test_import(import_data)
	df = import_data('./data/bank_data.csv')
	test_eda(perform_eda)
	test_encoder_helper(encoder_helper, df)


	df = encoder_helper(df, category_list, 'Churn')
	
	test_perform_feature_engineering(perform_feature_engineering, df, KEEP_COLS)
	X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X, y = perform_feature_engineering(
		df, KEEP_COLS)
	test_train_models(train_models,X,y)


'''
we could produce a report by storing each result in a list
then displaying a summary of all results

  passed_cases = len(list(filter(lambda x: x, result)))
    failed_cases = len(list(filter(lambda x: not x, result)))
    TOTAL_CASES = len(result)

    if all(result):
        # log success as final result
        logging.info("Final Test Result : Success %s/%s",
                     passed_cases, TOTAL_CASES
                     )
    else:
        # log failure as final result
        logging.error("Final Test Result : Failed %s/%s",
                      failed_cases, TOTAL_CASES
                      )


'''



