import numpy as np
import json

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, :-1]  # All columns except the last as features
    y = data[:, -1]   # Last column as labels
    return X, y

class MyXGBClassificationTree:
    def __init__(self, max_depth, reg_lambda, prune_gamma):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.prune_gamma = prune_gamma
        self.feature = None
        self.residual = None
        self.prev_yhat = None
        self.best_split_info = {}

    def x_predict(self, p, x):
        print(f"Current feature vector: {x}")  # Print the current feature vector
        if x[p['fid']] <= p['split_point']:
            print(f"Going left on feature {p['fid']} (value={x[p['fid']]} <= split={p['split_point']})")
            if isinstance(p['left'], dict):
                return self.x_predict(p['left'], x)
            else:
                return p['left']
        else:
            print(f"Going right on feature {p['fid']} (value={x[p['fid']]} > split={p['split_point']})")
            if isinstance(p['right'], dict):
                return self.x_predict(p['right'], x)
            else:
                return p['right']

    def output_value(self, did):
        r = self.residual[did]
        H = (self.prev_yhat[did] * (1. - self.prev_yhat[did])).sum()
        return np.sum(r) / (H + self.reg_lambda)

    def predict(self, x_test):
        p = self.best_split_info

        if isinstance(p, dict):
            y_pred = [self.x_predict(p, x) for x in x_test]
            return np.array(y_pred)
        else:
            return np.zeros(x_test.shape[0])  # Default predictions if no model is found

class MyXGBClassifier:
    def __init__(self):
        self.models = []
        self.eta = 0.3  # Default learning rate
        self.base_score = 0.5  # Initialize base_score

    def load_model_from_json(self, filepath):
        with open(filepath, 'r') as json_file:
            model_data = json.load(json_file)
            # Check if 'models' key exists, and load models
            if 'models' in model_data:
                max_depth = model_data['max_depth']
                reg_lambda = model_data['reg_lambda']
                prune_gamma = model_data['prune_gamma']

                for last_split_info in model_data['models']:
                    # Re-create each classification tree with the last split info
                    model = MyXGBClassificationTree(
                        max_depth=max_depth,
                        reg_lambda=reg_lambda,
                        prune_gamma=prune_gamma
                    )
                    model.best_split_info = last_split_info
                    self.models.append(model)

    def F2P(self, x):
        return 1. / (1. + np.exp(-x))

    def print_model_structure(self):
        if not self.models:
            print("No models are loaded.")
            return

        for i, model in enumerate(self.models):
            print(f"\nModel {i + 1} Structure:")
            print(f"Max Depth: {model.max_depth}")
            print(f"Regularization Lambda: {model.reg_lambda}")
            print(f"Pruning Gamma: {model.prune_gamma}")
            print("Best Split Info:", model.best_split_info)

    def predict(self, x_test, proba=False):
        Fm = np.zeros(shape=(x_test.shape[0],)) + self.base_score
        print(f"Initial Fm values: {Fm}")

        for model in self.models:
            model_predictions = model.predict(x_test)
            print(f"Model predictions: {model_predictions}")

            Fm += self.eta * model_predictions
            print(f"Updated Fm values: {Fm}")

        y_prob = self.F2P(Fm)

        print(f"Final probabilities: {y_prob}")

        if proba:
            return y_prob
        else:
            y_pred = (y_prob > 0.5).astype('uint8')
            return y_pred

# Load test data from the CSV file
test_filename = '/content/drive/MyDrive/project/data/class_test.csv'
X_test, _ = load_data(test_filename)

# Create an instance of the model
model = MyXGBClassifier()

# Load the trained model from JSON
model.load_model_from_json('best.json')

# Print model structure to verify
model.print_model_structure()

# Make predictions on the test data
predictions = model.predict(X_test)

# Print the predictions
print("\nPredictions on Test Data:")
print(predictions)
