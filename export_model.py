import mlflow.pyfunc
import mlflow

# Replace this with your MLflow tracking URI if needed
# mlflow.set_tracking_uri("http://your-mlflow-server:5000")

# Load model from MLflow registry (production stage)
model = mlflow.pyfunc.load_model("models:/Churn_Prediction_Model/production")

# Save it locally
mlflow.pyfunc.save_model(model, "models/Churn_Prediction_Model")
print("Model exported to local folder: models/Churn_Prediction_Model")
