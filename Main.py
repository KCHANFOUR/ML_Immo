from preprocess import X_train, X_test, y_train, y_test, X_transform, y
from flaskr import app
import test_model

if __name__=="__main__":
    print("Calculate result")
    app.app.run()

