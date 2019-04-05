from ludwig import LudwigModel
import pandas as pd

# import csv
data = pd.read_csv('dataset.csv', sep = ',')

# train a model
model_definition = {
        'input_features': [
                {'name': 'age', 'type': 'numerical', 'encoder': 'rnn'},
                {'name': 'sex', 'type': 'category', 'encoder': 'rnn'},
                {'name': 'trestbps', 'type': 'numerical', 'encoder': 'rnn'},
                {'name': 'chol', 'type': 'numerical', 'encoder': 'rnn'},
                {'name': 'fbs', 'type': 'numerical', 'encoder': 'rnn'},
                {'name': 'restecg', 'type': 'numerical', 'encoder': 'rnn'},
                {'name': 'thalach', 'type': 'numerical', 'encoder': 'rnn'},
                {'name': 'exang', 'type': 'binary', 'encoder': 'rnn'},
                {'name': 'oldpeak', 'type': 'numerical', 'encoder': 'rnn'},
                {'name': 'slope', 'type': 'category', 'encoder': 'rnn'},
                {'name': 'ca', 'type': 'category', 'encoder': 'rnn'},
                {'name': 'thal', 'type': 'category', 'encoder': 'rnn'}             
        ], 
        'output_features': [{'name': 'target', 'type': 'binary'}], 
        'training': {'epochs': 10}
        }
model = LudwigModel(model_definition)
train_stats = model.train(data)


# obtain predictions
predictions = model.predict(data)

model.close()