from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
digits = datasets.load_digits()


#Flattening the dataset
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.8, shuffle=False
)
clf = SVC(kernel="rbf")

pipe_svc = make_pipeline(PCA(n_components=2), clf)
pipe_svc.fit(X_train[:500], y_train[:500])

predicted = pipe_svc.predict(X_test[:100])

print(accuracy_score(y_test[:100], predicted))



