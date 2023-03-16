from churn_library import *

if __name__ == "__main__":


    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]

# columns used for X
    KEEP_COLS = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # model params
    params = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # importing data
    bank_data_df = import_data('./data/bank_data.csv')


    # generating eda charts
    perform_eda(bank_data_df)


    # encoding
    encoded_df = encoder_helper(bank_data_df, category_list, 'Churn')


    # data split
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X, y = perform_feature_engineering(
        encoded_df, KEEP_COLS)


    # train models with params
    Y_TRAIN_PREDS_LR, Y_TRAIN_PREDS_RF, Y_TEST_PREDS_LR, Y_TEST_PREDS_RF, CV_RFC = train_models(
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X, y, params)


    # logistic regression classification report
    classification_report_image(Y_TRAIN,
                                Y_TEST,
                                Y_TRAIN_PREDS_LR,
                                Y_TEST_PREDS_LR,
                                'Logistic Regression',
                                X_TRAIN,
                                X_TEST,)

    # random forest classification report
    classification_report_image(Y_TRAIN,
                                Y_TEST,
                                Y_TRAIN_PREDS_RF,
                                Y_TEST_PREDS_RF,
                                'Random Forest',
                                X_TRAIN,
                                X_TEST,)

    # feature importance
    feature_importance_plot(CV_RFC, X)
























