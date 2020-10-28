import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r"C:\Users\hanch\PycharmProjects\GlassViaMLP\Glass_dataset\glass.csv")
y = data["Type"]
X = data.drop(["Type"], axis=1)
numerical_features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
numerical_transformer = Pipeline(steps=[
    ("std_scaler", StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features)
])
def evaluate_classifier(label, classifier):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)
    print(label, ":", pipeline.score(X_test, y_test))
    y_pred = pipeline.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

# evaluate_classifier("MLP Classifier, hidden_layer_size=20",
#                    MLPClassifier(random_state=42, hidden_layer_sizes=(20,)))
# evaluate_classifier("MLP Classifier, hidden_layer_size=50",
#                    MLPClassifier(random_state=42, hidden_layer_sizes=(50,)))
evaluate_classifier("MLP Classifier, hidden_layer_size=100",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(100,)))
# evaluate_classifier("MLP Classifier, hidden_layer_size=20,10",
#                    MLPClassifier(random_state=42, hidden_layer_sizes=(20,10,)))
# evaluate_classifier("MLP Classifier, hidden_layer_size=50,20",
#                    MLPClassifier(random_state=42, hidden_layer_sizes=(50,20,)))
evaluate_classifier("MLP Classifier, hidden_layer_size=100,50",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(100,50,)))
evaluate_classifier("MLP Classifier, hidden_layer_size=100,50,20",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(100,50,20,)))
evaluate_classifier("MLP Classifier, hidden_layer_size=100,80,50,20",
                   MLPClassifier(random_state=42, hidden_layer_sizes=(100,80,50,20,)))