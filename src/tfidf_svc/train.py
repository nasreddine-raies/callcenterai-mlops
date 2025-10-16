import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import mlflow
print("Libraries imported successfully.")
# 1. Chargement des données
df = pd.read_csv('data/data_final.csv')

# 2. Préprocessing simple
df = df.dropna(subset=['Document', 'Topic_group'])
X = df['Document'].astype(str)
y = df['Topic_group'].astype(str)

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Entraînement SVM + Calibration
svc = LinearSVC(C=1.0, max_iter=1000)
clf = CalibratedClassifierCV(svc)  # Pour obtenir des probabilités
clf.fit(X_train_vec, y_train)

# 6. Évaluation
y_pred = clf.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print(classification_report(y_test, y_pred))

# 7. Sauvegarde des artefacts
joblib.dump(vectorizer, 'src/tfidf_svc/tfidf_vectorizer.joblib')
joblib.dump(clf, 'src/tfidf_svc/tfidf_svc_model.joblib')

# 8. MLflow tracking (optionnel)
mlflow.set_experiment("TFIDF_SVC")
with mlflow.start_run():
    mlflow.log_param("C", svc.C)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(clf, "tfidf_svc_model")
    mlflow.log_artifact("src/tfidf_svc/tfidf_vectorizer.joblib")
    mlflow.log_artifact("src/tfidf_svc/tfidf_svc_model.joblib")