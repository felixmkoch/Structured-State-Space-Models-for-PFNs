from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, MambaPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

#mamba_model_path = "models_diff/mamba_custom.cpkt"
transformer_model_path = 'models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'

#classifier = MambaPFNClassifier(device='cuda', N_ensemble_configurations=32, model_path=mamba_model_path)
classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=32, model_path=transformer_model_path)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))