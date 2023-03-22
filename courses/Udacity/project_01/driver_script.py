from churn_library import *

if __name__ == "__main__":
 
    # importing data
    bank_data_df = import_data('./data/bank_data.csv')
    print(bank_data_df.head(5))

    # generating eda charts
    perform_eda(bank_data_df)

    # encoding
    encoded_df = encoder_helper(bank_data_df, category_list, 'Churn')
    print(encoded_df.head(5))

    # data split
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X, y = perform_feature_engineering(
        encoded_df, KEEP_COLS)
    print(X_TRAIN.head(5))
    print(X_TEST.head(5))
    print(Y_TRAIN.head(5))
    print(Y_TEST.head(5))
    print(X.head(5))
    print(y.head(5))


    # train models with params
    Y_TRAIN_PREDS_LR, Y_TRAIN_PREDS_RF, Y_TEST_PREDS_LR, Y_TEST_PREDS_RF, CV_RFC = train_models(
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X, y, params)
    print(Y_TRAIN_PREDS_LR)
    print(Y_TRAIN_PREDS_RF)
    print(Y_TEST_PREDS_LR)
    print(Y_TEST_PREDS_RF)
    print(CV_RFC)


    # logistic regression classification report
    classification_report_image(Y_TRAIN,
                                Y_TEST,
                                Y_TRAIN_PREDS_LR,
                                Y_TEST_PREDS_LR,
                                Y_TRAIN_PREDS_RF,
                                Y_TEST_PREDS_RF,
                                X_TRAIN,
                                X_TEST,)

    # # random forest classification report
    # classification_report_image(Y_TRAIN,
    #                             Y_TEST,
    #                             Y_TRAIN_PREDS_RF,
    #                             Y_TEST_PREDS_RF,
    #                             'Random Forest',
    #                             X_TRAIN,
    #                             X_TEST,)

    # feature importance
    feature_importance_plot(CV_RFC, X)


















