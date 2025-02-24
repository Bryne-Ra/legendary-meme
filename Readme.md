# Starbucks Analytics Project

## Overview
The **Starbucks Analytics Project** is a data-driven application designed to analyze user behavior, demographics, and engagement with Starbucks digital campaigns. It leverages machine learning to predict offer completion rates and provides interactive visualizations to explore user data. The project integrates data cleaning, preprocessing, model training, and a Flask-based web dashboard for visualization and prediction.

This project utilizes **Python, Flask, SQLite, Plotly, and machine learning libraries (scikit-learn, XGBoost)** to process and analyze Starbucks customer data, stored in JSON and CSV files, and saved to a SQLite database for persistence.

## Key Features
- **Data Processing:** Loads and cleans data from JSON files (*transcript.json, profile.json, portfolio.json*) or a SQLite database, handling missing values, sentinel values (*e.g., age = 118*), and data merging.
- **Machine Learning:** Trains and evaluates three models (*Logistic Regression, Random Forest, XGBoost*) to predict offer completion, saving the best model to a pickle file for use in the Flask app.
- **Visualizations:** Generates six interactive **Plotly** visualizations:
  1. Age Distribution
  2. Gender Distribution
  3. Income Distribution
  4. Offer Completion by Gender
  5. Offer Completion by Age Group
  6. Offer Completion by Income Group
- **Web Dashboard:** A Flask-based website with routes for an analytics dashboard, Starbucks analytics content, and predictions, displaying visualizations and offering real-time predictions based on user input.
- **Data Storage:** Saves cleaned data to a SQLite database (*cleaned_data.db*) for efficient querying and persistence.

## Prerequisites
Before running the project, ensure you have the following installed:

- **Python 3.8 or higher**
- **Required Python libraries:**
  ```bash
  pip install pandas numpy plotly flask sqlalchemy joblib scikit-learn xgboost
  ```

## Project Structure
```
Starbucks Project Workspace/
├── app/
│   ├── run.py              # Flask application with routes, predictions, and feature importance
│   ├── visualization.py    # Visualization generation functions using Plotly
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # CSS for styling the web dashboard
│   │   └── images/
│   │       └── starbucks_logo_resized.png  # Logo for the dashboard
│   └── templates/
│       ├── master.html     # Dashboard template
│       ├── starbucks_analytics.html  # Static content template
│       └── predict.html    # Prediction template
├── data/
│   ├── final_df.csv        # Processed CSV data for visualizations
│   ├── transcript.json     # Transaction data
│   ├── profile.json        # Profile data
│   ├── portfolio.json      # Portfolio data
│   ├── cleaned_data.db     # SQLite database for cleaned data
├── models/
│   └── best_model.pkl      # Saved machine learning model
└── README.md               # This file
```

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Bryne-Ra/starbucks-analytics.git
   cd starbucks-analytics
   ```
2. **Set Up the Environment:**
   - Install the required Python libraries using the command above.
3. **Prepare Data Files:**
   - Place *transcript.json, profile.json, and portfolio.json* in the `data/` directory.
   - Ensure `final_df.csv` is available in `data/` for visualizations (or generate it using the data processing script).
4. **Set Up SQLite Database:**
   - Run the data processing script to clean and save data to `cleaned_data.db`:
     ```bash
     python data_processing.py transcript.json profile.json portfolio.json cleaned_data.db
     ```
5. **Train the Model:**
   - Run the model training script to train and save the best model to `models/best_model.pkl`:
     ```bash
     python train_model.py ../data/cleaned_data.db ../models/best_model.pkl
     ```
6. **Run the Flask App:**
   - Navigate to the `app/` directory.
   - Start the Flask application:
     ```bash
     python3 run.py ../models/best_model.pkl ../data/cleaned_data.db
     ```
   - Open your browser and visit `http://127.0.0.1:5000/` to access the dashboard.

## Usage
### Dashboard
Visit `http://127.0.0.1:5000/` to view the **Analytics Dashboard**, which displays six interactive visualizations:
- Age Distribution
- Gender Distribution
- Income Distribution
- Offer Completion by Gender
- Offer Completion by Age Group
- Offer Completion by Income Group

Navigate using the header menu to:
- **Starbucks Analytics (`/starbucks-analytics`)**: View a static article about user behavior and demographics.
- **Predict (`/predict`)**: Access a form to predict offer completion based on user input (*age, income, gender, channels, offer type*), displaying feature importance and prediction results.

### Data Processing
- Use `data_processing.py` to load, clean, and save data from JSON files to `cleaned_data.db`.
- The script merges transaction, profile, and portfolio data, handles missing values (*e.g., age = 118, empty strings*), and calculates derived metrics like `days_to_complete`.

### Model Training
- Use `model_training.py` to train Logistic Regression, Random Forest, and XGBoost models on the cleaned data, selecting the best model based on accuracy and saving it to `models/best_model.pkl`.
- The model predicts whether a user will complete an offer based on features like *age, income, gender, channels, and offer type*.

### Visualizations
- `visualization.py` generates **Plotly** visualizations, which are rendered in the Flask app’s dashboard using `master.html`.
- Visualizations are interactive, allowing users to hover, zoom, and explore data dynamically.

## Configuration
- **File Paths:**
  - Update `TRANSACTION_DATA_PATH`, `PROFILE_DATA_PATH`, `PORTFOLIO_DATA_PATH`, `DATABASE_PATH`, `MODEL_OUTPUT_PATH`, and `data_path` in the scripts if your file locations differ.
- **Dependencies:**
  - Ensure all required libraries are installed as specified in **Prerequisites**.
- **Environment:**
  - The Flask app runs on `http://127.0.0.1:5000/` by default in debug mode. Adjust `app.run()` in `run.py` for production or different ports.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make changes and test thoroughly.
4. Submit a pull request with a description of your changes.

## License
This project is licensed under the **MIT License** - see the `LICENSE` file for details.

## Contact
For questions or support, contact the project maintainer:
- **Email:** brianrathabe@gmail.com
- **GitHub:** [Bryne-Ra](https://github.com/Bryne-Ra)

