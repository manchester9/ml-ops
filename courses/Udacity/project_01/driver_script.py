from churn_library import *

if __name__ == "__main__":
 
    # importing data
    df = import_data('./data/bank_data.csv')
    print(df.head(5))

    # generating eda charts
    perform_eda(df)

    # encoding
    encoded_df = encoder_helper(df, category_list, 'Churn')
    print(encoded_df.head(5))

    # data split
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X, y = perform_feature_engineering(
        encoded_df, KEEP_COLS)

    # train models with params
    Y_TRAIN_PREDS_LR, Y_TRAIN_PREDS_RF, Y_TEST_PREDS_LR, Y_TEST_PREDS_RF, CV_RFC = train_models(
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X, y, params)

    # classification report
    classification_report_image(Y_TRAIN,
                                Y_TEST,
                                Y_TRAIN_PREDS_LR,
                                Y_TEST_PREDS_LR,
                                Y_TRAIN_PREDS_RF,
                                Y_TEST_PREDS_RF,
                                X_TRAIN,
                                X_TEST,)

    # feature importance
    feature_importance_plot(CV_RFC, X)


















