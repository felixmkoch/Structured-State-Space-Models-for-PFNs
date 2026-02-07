from hydrapfn.scripts import tabular_metrics


def get_scoring_string(metric_used, multiclass=True, usage="sklearn_cv"):
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        if usage == 'sklearn_cv':
            return 'roc_auc_ovo'
        elif usage == 'autogluon':
            #return 'log_loss' # Autogluon crashes when using 'roc_auc' with some datasets usning logloss gives better scores;
                              # We might be able to fix this, but doesn't work out of box.
                              # File bug report? Error happens with dataset robert and fabert
            if multiclass:
                return 'roc_auc_ovo_macro'
            else:
                return 'roc_auc'
        elif usage == 'tabnet':
            return 'logloss' if multiclass else 'auc'
        elif usage == 'autosklearn':
            import autosklearn.classification
            if multiclass:
                return autosklearn.metrics.log_loss # roc_auc only works for binary, use logloss instead
            else:
                return autosklearn.metrics.roc_auc
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss, ROC not available
        elif usage == 'xgb':
            return 'logloss'
        elif usage == 'lightgbm':
            if multiclass:
                return 'auc'
            else:
                return 'binary'
        return 'roc_auc'
    elif metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        if usage == 'sklearn_cv':
            return 'neg_log_loss'
        elif usage == 'autogluon':
            return 'log_loss'
        elif usage == 'tabnet':
            return 'logloss'
        elif usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.log_loss
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss
        return 'logloss'
    elif metric_used.__name__ == tabular_metrics.r2_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.r2
        elif usage == 'sklearn_cv':
            return 'r2' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'r2'
        elif usage == 'xgb': # XGB cannot directly optimize r2
            return 'rmse'
        elif usage == 'catboost': # Catboost cannot directly optimize r2 ("Can't be used for optimization." - docu)
            return 'RMSE'
        else:
            return 'r2'
    elif metric_used.__name__ == tabular_metrics.root_mean_squared_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.root_mean_squared_error
        elif usage == 'sklearn_cv':
            return 'neg_root_mean_squared_error' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'rmse'
        elif usage == 'xgb':
            return 'rmse'
        elif usage == 'catboost':
            return 'RMSE'
        else:
            return 'neg_root_mean_squared_error'
    elif metric_used.__name__ == tabular_metrics.mean_absolute_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.mean_absolute_error
        elif usage == 'sklearn_cv':
            return 'neg_mean_absolute_error' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'mae'
        elif usage == 'xgb':
            return 'mae'
        elif usage == 'catboost':
            return 'MAE'
        else:
            return 'neg_mean_absolute_error'
        
    elif metric_used.__name__ == tabular_metrics.accuracy_metric.__name__:
        return 'acc'
    else:
        raise Exception('No scoring string found for metric')