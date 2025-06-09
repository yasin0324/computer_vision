# Plant Disease Recognition System

## 🧹 Project Cleanup

A number of cleanup activities have been performed to improve the project structure and remove redundant files.

### Deleted Files

1.  **System Files**
    -   `.DS_Store` (macOS system files)
    -   `src/.DS_Store`
2.  **Duplicate Scripts**
    -   `scripts/test_model.py` (Duplicate test script)
    -   `scripts/test_visualization.py` (Duplicate visualization test)
    -   `scripts/evaluate_baseline.py` (Replaced by `comprehensive_evaluation.py`)
    -   `scripts/quick_train.py` (Old version)

### Core Scripts Retained

-   `scripts/train_baseline_improved.py` - Improved baseline model training
-   `scripts/train_se_net.py` - SE-Net model training
-   `scripts/train_cbam.py` - CBAM model training
-   `scripts/comprehensive_evaluation.py` - Comprehensive evaluation script
-   `scripts/evaluation_example.py` - Evaluation example
-   `scripts/analyze_data_distribution.py` - Data analysis
-   `scripts/validate_setup.py` - Environment validation

## 🔧 Web Application

A web application is available to interact with the models.

### Features

1.  **ModelPredictor**:
    -   Uses a mock model for demonstration.
    -   Supports simulated prediction results.
    -   Includes error handling.
2.  **TrainingManager**:
    -   Simulates the training process.
    -   Provides real-time status updates and progress monitoring.
3.  **FileManager**:
    -   Scans the file system.
    -   Provides dataset statistics.
    -   Keeps a history of evaluations.

### Web Application Access

-   **Home**: `http://localhost:5000`
-   **Disease Recognition**: `http://localhost:5000/predict`
-   **Model Training**: `http://localhost:5000/train`
-   **Model Evaluation**: `http://localhost:5000/evaluate`
-   **Model Comparison**: `http://localhost:5000/compare`
-   **System Dashboard**: `http://localhost:5000/dashboard`

### API Endpoints

-   `GET /api/models` - Get list of available models
-   `POST /api/predict` - Image prediction
-   `POST /api/train` - Start training
-   `GET /api/training_status/<id>` - Training status
-   `POST /api/evaluate` - Model evaluation
-   `POST /api/compare` - Model comparison
-   `GET /api/dashboard_data` - Dashboard data

## 📁 Project Structure

```
.
├── src/                    # Core source code
│   ├── models/             # Model definitions
│   ├── data/               # Data processing
│   ├── training/           # Training modules
│   ├── evaluation/         # Evaluation modules
│   ├── config/             # Configuration files
│   └── utils/              # Utility functions
├── webapp/                 # Web application
│   ├── app.py              # Flask main application
│   ├── utils.py            # Web utilities
│   ├── templates/          # HTML templates
│   └── requirements.txt    # Web dependencies
├── scripts/                # Scripts
├── data/                   # Datasets
├── models/                 # Trained models
├── outputs/                # Outputs
├── logs/                   # Logs
├── docs/                   # Documentation
├── tests/                  # Tests
└── notebooks/              # Jupyter notebooks
```

## 🎯 Next Steps

### 1. Model Training

-   Use `scripts/train_baseline_improved.py` to train a baseline model.
-   Once trained, the real prediction functionality can be used.

### 2. Feature Expansion

-   Add more data augmentation strategies.
-   Implement attention mechanism visualization.
-   Add model performance comparison charts.

### 3. Deployment Optimization

-   Configure a production-grade WSGI server.
-   Add user authentication and permission management.
-   Optimize frontend performance and user experience.
