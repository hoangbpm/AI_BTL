import numpy as np  
import json 
import os 

# Function to load data from CSV files  
def load_data(filename):  
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)  
    X = data[:, :-1]  
    y = data[:, -1]  
    return X, y  

# Function to log the training loss for a specific version  
def log_training_loss(version_folder, filename, loss_history):  
    log_filename = os.path.join(version_folder, 'log_train.txt')  
    with open(log_filename, 'a') as f:  
        loss_str = ', '.join(map(str, loss_history))  # Convert the loss history to a comma-separated string  
        f.write(f'File: {filename}, Loss: [{loss_str}]\n')  

# Function to log accuracy results for a specific version  
def log_accuracy(version_folder, filename, accuracy):  
    result_filename = os.path.join(version_folder, 'log_accuracy.txt')  
    with open(result_filename, 'a') as f:  
        f.write(f'File: {filename}, Accuracy: {accuracy:.2f}%\n')  

        
# Decision tree class  
class MyXGBClassificationTree:  
    def __init__(self, max_depth, reg_lambda, prune_gamma):  
        self.max_depth = max_depth  
        self.reg_lambda = reg_lambda  
        self.prune_gamma = prune_gamma  
        self.feature = None  
        self.residual = None  
        self.prev_yhat = None  
        self.best_split_info = {}  

    def node_split(self, did):  
        r = self.reg_lambda  
        max_gain = -np.inf  
        d = self.feature.shape[1]  
        G = self.residual[did].sum()  
        H = (self.prev_yhat[did] * (1. - self.prev_yhat[did])).sum()  
        p_score = (G ** 2) / (H + r)  

        for k in range(d):  
            GL = HL = 0.0  
            x_feat = self.feature[did, k]  
            x_uniq = np.unique(x_feat)  
            s_point = [np.mean([x_uniq[i - 1], x_uniq[i]]) for i in range(1, len(x_uniq))]  
            l_bound = -np.inf  
            for j in s_point:  
                left = did[np.where(np.logical_and(x_feat > l_bound, x_feat <= j))[0]]  
                right = did[np.where(x_feat > j)[0]]  

                GL += self.residual[left].sum()  
                HL += (self.prev_yhat[left] * (1. - self.prev_yhat[left])).sum()  
                GR = G - GL  
                HR = H - HL  

                gain = (GL ** 2) / (HL + r) + (GR ** 2) / (HR + r) - p_score  

                if gain > max_gain:  
                    max_gain = gain  
                    b_fid = k  
                    b_point = j  
                l_bound = j  

        if max_gain >= self.prune_gamma:  
            x_feat = self.feature[did, b_fid]  
            b_left = did[np.where(x_feat <= b_point)[0]]  
            b_right = did[np.where(x_feat > b_point)[0]]  
            self.best_split_info = {  
                'feature_index': b_fid,  
                'split_point': b_point,  
                'gain': max_gain,  
                'left_indices': b_left.tolist(),  
                'right_indices': b_right.tolist()  
            }  
            print(f"Best split found on feature {b_fid} at point {b_point} with gain {max_gain}")  
            return {'fid': b_fid, 'split_point': b_point, 'gain': max_gain,  
                    'left': b_left, 'right': b_right}  
        else:  
            print(f"No significant split found. Max Gain: {max_gain}.")  
            return None  

    def recursive_split(self, node, curr_depth):  
        left = node['left']  
        right = node['right']  

        if curr_depth >= self.max_depth:  
            return  

        s = self.node_split(left)  
        if isinstance(s, dict):  
            node['left'] = s  
            self.recursive_split(node['left'], curr_depth + 1)  

        s = self.node_split(right)  
        if isinstance(s, dict):  
            node['right'] = s  
            self.recursive_split(node['right'], curr_depth + 1)  

    def output_value(self, did):  
        r = self.residual[did]  
        H = (self.prev_yhat[did] * (1. - self.prev_yhat[did])).sum()  
        return np.sum(r) / (H + self.reg_lambda)  

    def output_leaf(self, d):  
        if isinstance(d, dict):  
            for key, value in d.items():  
                if key == 'left' or key == 'right':  
                    rtn = self.output_leaf(value)  
                    if rtn[0] == 1:  
                        d[key] = rtn[1]  
            return 0, 0  
        else:  
            return 1, self.output_value(d)  

    def fit(self, x, y, prev_yhat):  
        self.feature = x  
        self.residual = y  
        self.prev_yhat = prev_yhat  

        root = self.node_split(np.arange(x.shape[0]))  
        if isinstance(root, dict):  
            self.recursive_split(root, curr_depth=1)  

        if isinstance(root, dict):  
            self.output_leaf(root)  
            self.best_split_info = root  
            return root  

        return None  

    def x_predict(self, p, x):  
        if x[p['fid']] <= p['split_point']:  
            if isinstance(p['left'], dict):  
                return self.x_predict(p['left'], x)  
            else:  
                return p['left']  
        else:  
            if isinstance(p['right'], dict):  
                return self.x_predict(p['right'], x)  
            else:  
                return p['right']  

    def predict(self, x_test):  
        p = self.best_split_info  

        if isinstance(p, dict):  
            y_pred = [self.x_predict(p, x) for x in x_test]  
            return np.array(y_pred)  
        else:  
            return self.prev_yhat * x_test.shape[0]  

# Main classification class  
class MyXGBClassifier:  
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.3,  
                 prune_gamma=0.0, reg_lambda=0.0, base_score=0.5):  
        self.n_estimators = n_estimators  
        self.max_depth = max_depth  
        self.eta = learning_rate  
        self.prune_gamma = prune_gamma  
        self.reg_lambda = reg_lambda  
        self.base_score = base_score  
        self.models = []  
        self.loss = []  
        self.split_info_history = []  

    def F2P(self, x):  
        return 1. / (1. + np.exp(-x))  

    def fit(self, x, y,z):  
        # Calculate initial value  
        F0 = np.log(self.base_score / (1. - self.base_score))  
        Fm = np.repeat(F0, x.shape[0])  
        y_hat = self.F2P(Fm)  

        for m in range(self.n_estimators):  
            print(f"\nFitting estimator {m + 1}/{self.n_estimators}")  
            residual = y - y_hat  

            model = MyXGBClassificationTree(max_depth=self.max_depth,  
                                            reg_lambda=self.reg_lambda,  
                                            prune_gamma=self.prune_gamma)  
            model.fit(x, residual, y_hat)  

            if model.best_split_info:  
                self.split_info_history.append(model.best_split_info)  

            gamma = model.predict(x)  
            Fm += self.eta * gamma  
            y_hat = self.F2P(Fm)  

            self.models.append(model)  

            loss_value = (-(y * np.log(y_hat + 1e-8) + (1. - y) * np.log(1. - y_hat + 1e-8)).sum())/z
            self.loss.append(loss_value)  
            print(f"Loss after estimator {m + 1}: {loss_value}")  

        # Save all model information to a single file `best.json`  
        self.save_all_models_to_json('best.json')  

        return self.loss  

    def save_all_models_to_json(self, filepath):  
        model_data = {  
            'n_estimators': self.n_estimators,  
            'max_depth': self.max_depth,  
            'learning_rate': self.eta,  
            'prune_gamma': self.prune_gamma,  
            'reg_lambda': self.reg_lambda,  
            'base_score': self.base_score,  
            'models': []  # List to hold all model information  
        }  

        for model in self.models:  
            model_data['models'].append(model.best_split_info)  # Save info for each model  

        with open(filepath, 'w') as json_file:  
            json.dump(model_data, json_file, indent=4)  

    def predict(self, x_test, proba=False):  
        Fm = np.zeros(shape=(x_test.shape[0],)) + self.base_score  
        for model in self.models:  
            Fm += self.eta * model.predict(x_test)  

        y_prob = self.F2P(Fm)  

        if proba:  
            return y_prob  
        else:  
            y_pred = (y_prob > 0.5).astype('uint8')  
            return y_pred  


if __name__ == "__main__":  
    # Version input  
    version = input("Enter version number (e.g., 1, 2, ...): ")  
    
    # Create a folder for models if it doesn't exist  
    models_dir = 'models'  
    os.makedirs(models_dir, exist_ok=True)  

    # Create a folder for the specific version  
    version_folder = os.path.join(models_dir, f'v{version}')  
    os.makedirs(version_folder, exist_ok=True)  

    # List of input CSV files  
    filenames = [  
        r'local\class_0.csv',  
        r'local\class_1.csv',    
        r'local\class_2.csv',    
    ]  

    for i, filename in enumerate(filenames):  
        # Load data from the current file  
        X, y = load_data(filename)  
        split_index = int(0.8 * len(X))  # Calculate the index for splitting  
        X_train, X_val = X[:split_index], X[split_index:]  # Split features  
        y_train, y_val = y[:split_index], y[split_index:]  # Split labels
        num_X_train = len(X_train)  

        # Initialize and train the model for the current dataset  
        model = MyXGBClassifier(n_estimators=15, max_depth=6, learning_rate=0.3, prune_gamma=0.0)  
        loss_history = model.fit(X_train, y_train,num_X_train)  

        # Print loss history  
        print(f"\nLoss History for model trained on {filename}:")  
        print(loss_history)  

        # Predict on validation data  
        predictions = model.predict(X_val)  

        # Calculate accuracy  
        accuracy = np.mean(predictions == y_val) * 100  # Calculate accuracy as percentage  

        print(f"\nAccuracy for model trained on {filename}: {accuracy:.2f}%")  
        print("\nActual y_val:")  
        print(y_val.astype(int))  # Convert to integer type  
        print("\nPredictions on validation data:")  
        print(predictions.astype(int))  # Optionally, convert predictions to integer as well  

        # Log training loss  
        log_training_loss(version_folder, filename, loss_history)  

        # Log accuracy results  
        log_accuracy(version_folder, filename, accuracy)  

        # Print the best split information from all models  
        print("\nBest Split Information from All Models:")  
        for j, split_info in enumerate(model.split_info_history):  
            print(f"Model {j + 1}:")  
            print(split_info)  

        # Save model weights as JSON files with versioning  
        model.save_all_models_to_json(os.path.join(version_folder, f'model_{i + 1}.json'))  
