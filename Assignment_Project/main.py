# main.py
from data_loader import load_data, show_missing
from model import preprocess, create_model
from train import train_model, evaluate_model

if __name__ == "__main__":
    path = "/home/inventor/Machine_learning/Assignment_Project/Data/AmesHousing.csv"
    df = load_data(path)
    show_missing(df)
    
    X_train, X_test, y_train, y_test = preprocess(df)
    model = create_model()
    
    trained_model = train_model(model, X_train, y_train)
    r2, mse = evaluate_model(trained_model, X_test, y_test)
    
    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}")
