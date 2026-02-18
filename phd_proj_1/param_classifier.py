from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sn
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# function to evaluate each model with different metrics
def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate evaluation metrics for a classifier.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for reporting

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, F1-score,
        and balanced accuracy
    """
    results = {
        'Model': model_name,
        'Accuracy': np.round(accuracy_score(y_true, y_pred), 4),
        'Balanced Accuracy': np.round(balanced_accuracy_score(y_true, y_pred), 4),
        'Precision': np.round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'Recall': np.round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'F1-Score': np.round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4)
    }
    return results

# results list to store evaluation metrics for each model based on the above function
results_list = []

# function for plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    """Plot confusion matrix with consistent styling."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 10))
    sn.set(font_scale=0.6)
    sn.heatmap(cm, annot=True, annot_kws={'size': 6},
               cmap=plt.cm.Greens, linewidths=0.1, fmt='g')

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks + 0.5, labels, rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, format='png', dpi=150, bbox_inches='tight')
    plt.show()

# Model hyperparameters
DT_MAX_DEPTH = 18
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 16
RF_MIN_SAMPLES_LEAF = 3
ERT_N_ESTIMATORS = 200
ERT_MIN_SAMPLES_LEAF = 5
XGB_N_ESTIMATORS = 30
XGB_MAX_DEPTH = 10
XGB_LEARNING_RATE = 0.3
XGB_N_JOBS = 16

# Data configuration
TEST_SIZE = 0.25
FEATURES = ['Voc', 'jsc', 'ff', 'mpce']
TARGET = 'sweeps'

def main():
    """Main execution function."""
    # df = pd.read_csv(filedir+r'\proj_1_data_table_c_n')
    df = pd.read_csv(os.path.join(DATA_DIR, 'proj_1_data_table_c_n.csv'))

    pd.set_option('display.max_columns', None)
    print(df.head(2))

    #this if you are not interested in seeing the distribution of the sweeping parameters
    print(df['sweeps'].value_counts())

    # Feature set used in the paper
    new_df = df[['Voc', 'jsc', 'ff','mpce','sweeps']]
    # Alternative feature sets to explore:
    # - including currents at 0.5 V and 1 V: new_df = df[['Voc', 'jsc', 'ff', 'j_05','j_1','mpce','sweeps']]
    # - without mpce: new_df = df[['Voc', 'jsc', 'ff','sweeps']]

    # input features are all columns except the last one in the new data frame,
    # targets/sweeping params are in the last column of the new data frame
    t_features = new_df.columns[0:-1]
    # feature_labels = new_df.columns.values[0:-1]
    # print('feature_labels: ', feature_labels)
    t_targets = new_df.columns[-1]

    # train and test split, 25% test size, shuffle the data
    X_train, X_test, y_train, y_test = train_test_split(new_df[t_features],
                new_df[t_targets], test_size=TEST_SIZE, random_state=None, shuffle=True)

    # get labels, which are the unique sweeping parameters
    labels = sorted(df[TARGET].unique())

    # transform to float32, scikit cannot handle float64
    # actually, this is not necessary anymore. probably due to an update in scikit-learn.
    # if you run into an error, try uncommenting these lines
    # X_train = np.nan_to_num(X_train.astype(np.float32))
    # X_test = np.nan_to_num(X_test.astype(np.float32))

    # DT classifier
    clt = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH)
    clt.fit(X_train, y_train)

    # prediction
    y_pred_dt = clt.predict(X_test)
    results_list.append(evaluate_model(y_test, y_pred_dt, 'Decision Tree'))

    # prediction probabilities
    clt.predict_proba(X_test)

    # feature imortance
    fif = pd.DataFrame(clt.feature_importances_,
                      index=new_df.columns.values[0:-1],
                      columns=['importance'])
    fi_sorted_dt = fif.sort_values('importance', ascending=False)

    plot_confusion_matrix(y_test, y_pred_dt, labels, 'Decision Tree')



    # Random Forest
    clr = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,max_depth=RF_MAX_DEPTH,random_state=None,
                                 bootstrap=True, min_samples_leaf=RF_MIN_SAMPLES_LEAF)
    clr.fit(X_train, y_train)

    # prediction
    y_pred_rf = clr.predict(X_test)
    results_list.append(evaluate_model(y_test, y_pred_rf, 'Random Forest'))

    # feature imortance
    fir = pd.DataFrame(clr.feature_importances_,
                      index=new_df.columns.values[0:-1],
                      columns=['importance'])
    fi_sorted_rf = fir.sort_values('importance', ascending=False)

    plot_confusion_matrix(y_test, y_pred_rf, labels, 'Random Forest')


    # extra tree classifier = extremely randomized forest
    ert = ExtraTreesClassifier(n_estimators=ERT_N_ESTIMATORS,max_depth=None,random_state=0,bootstrap=True,
                               min_samples_leaf=ERT_MIN_SAMPLES_LEAF)
    ert.fit(X_train,y_train)

    # prediction
    y_pred_ert = ert.predict(X_test)
    results_list.append(evaluate_model(y_test, y_pred_ert, 'Extra Trees'))

    # feature imortance
    fiert = pd.DataFrame(ert.feature_importances_,
                      index=new_df.columns.values[0:-1],
                      columns=['importance'])
    fi_sorted_ert = fiert.sort_values('importance', ascending=False)

    plot_confusion_matrix(y_test, y_pred_ert, labels, 'Extremely Randomized Tree')

    # XGBoost

    # create label encoder for XGBoost
    label_encoder = LabelEncoder()
    label_encoder.fit(new_df[t_targets])

    # transform string labels to numeric for XGBoost
    y_train_encoded = label_encoder.transform(y_train)

    xgbm = xgb.XGBClassifier(booster='gbtree', n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
        n_jobs=XGB_N_JOBS, random_state=0, eval_metric='mlogloss')

    xgbm.fit(X_train, y_train_encoded)

    # prediction
    y_pred_xgb_encoded = xgbm.predict(X_test)

    # transform predictions back to original string labels
    y_pred_xgb = label_encoder.inverse_transform(y_pred_xgb_encoded)
    results_list.append(evaluate_model(y_test, y_pred_xgb, 'XGBoost'))


    # feature imortance
    fixgb = pd.DataFrame(xgbm.feature_importances_,
                      index=new_df.columns.values[0:-1],
                      columns=['importance'])
    fi_sorted_xgb = fixgb.sort_values('importance', ascending=False)

    plot_confusion_matrix(y_test, y_pred_xgb, labels, 'XGBoost')


    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('F1-Score', ascending=False)

    print("\n" + "="*80)
    print("model comparison")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    # bar diagram for metric comparison
    sn.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['Accuracy', 'Balanced Accuracy', 'F1-Score', 'Precision']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_df_sorted = results_df.sort_values(metric, ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df_sorted)))
        ax.barh(results_df_sorted['Model'], results_df_sorted[metric], color=colors)
        ax.set_title(metric)
        ax.set_xlim([0, 1])
        ax.grid(False)

        # show values at bars
        for i, v in enumerate(results_df_sorted[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, model_comparison.png', format='png', dpi=150, bbox_inches='tight')
    plt.show()

    # detailed classification reports
    print("\n" + "="*80)
    print("Model reports")
    print("="*80)

    models = {
        'Decision Tree': y_pred_dt,
        'Random Forest': y_pred_rf,
        'Extra Trees': y_pred_ert,
        'XGBoost': y_pred_xgb
    }

    for model_name, predictions in models.items():
        print(f"\n{model_name}:")
        print("-" * 80)
        print(classification_report(y_test, predictions, labels=labels, zero_division=0))

    # compare feature importances
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    importance_dfs = [
        (fi_sorted_dt, 'Decision Tree'),
        (fi_sorted_rf, 'Random Forest'),
        (fi_sorted_ert, 'Extra Trees'),
        (fi_sorted_xgb, 'XGBoost')
    ]

    for idx, (fi_df, title) in enumerate(importance_dfs):
        ax = axes[idx // 2, idx % 2]
        fi_df.plot(kind='barh', ax=ax, legend=False, color='steelblue')
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.invert_yaxis()
        ax.grid(False)

    plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, feature_importance_comparison.png', format='png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()


