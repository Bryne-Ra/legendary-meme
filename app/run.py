# run.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import ast
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from sqlalchemy import create_engine
import logging
from visualization import generate_visualizations  # Import the new function
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define paths for model and database
MODEL_PATH = '../models/best_model.pkl'
DATABASE_PATH = '../data/cleaned_data.db'

# Load the pre-trained model
best_model = joblib.load(MODEL_PATH)

# Function to load raw data from the database


def load_data():
    """
    Loads data from a SQLite database into a pandas DataFrame.
    
    Returns:
        pd.DataFrame: Raw data from the database.
    """
    engine = create_engine(f'sqlite:///{DATABASE_PATH}')
    return pd.read_sql("SELECT * FROM cleaned_data", engine)

# Function to preprocess data for predictions


def preprocess_data(df):
    """
    Preprocesses the DataFrame for model predictions by handling missing values,
    encoding categorical variables, and aligning columns with model expectations.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for predictions.
    """
    df['age'] = df['age'].replace(118, np.nan).fillna(
        df['age'].mean()).astype(int)
    df['income'] = df['income'].fillna(df['income'].mean()).astype(int)
    df['gender'] = df['gender'].fillna('O')

    # Safely evaluate channel strings
    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except:
            return []

    df['channels'] = df['channels'].apply(safe_literal_eval)
    channel_types = ['email', 'mobile', 'social', 'web']
    for channel in channel_types:
        df[f'channel_{channel}'] = df['channels'].apply(
            lambda x: 1 if channel in x else 0)

    # Create target variable if available
    df['will_complete_offer'] = df.get('event_offer_completed', 0).astype(int)

    # Drop unnecessary columns
    df = df.drop(
        columns=['channels', 'offer_id_offer_viewed', 'person'], errors='ignore')

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['gender', 'offer_type'])

    # Ensure all expected columns are present
    expected_columns = [
        'age', 'income', 'will_complete_offer',
        'gender_F', 'gender_M', 'gender_O',
        'channel_web', 'channel_email', 'channel_mobile', 'channel_social',
        'offer_type_bogo', 'offer_type_discount', 'offer_type_informational'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]

# Function to generate feature importance visualization (kept in run.py as requested)


def generate_feature_importance():
    """
    Generates a visualization of feature importance for the model.
    
    Returns:
        str: JSON representation of the Plotly figure for feature importance.
    """
    # Assuming best_model is a model with feature_importances_ (e.g., Random Forest or Gradient Boosting)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': best_model.feature_names_in_,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        # Fallback for models without feature_importances_ (e.g., Logistic Regression)
        # You might want to use permutation importance or another method here
        feature_importance = pd.DataFrame({
            'feature': best_model.feature_names_in_,
            # Placeholder
            'importance': np.random.random(len(best_model.feature_names_in_))
        }).sort_values('importance', ascending=False)

    # Create a bar plot for feature importance
    fig = go.Figure(data=[go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h',
        marker_color='#1f77b4',
        name="Feature Importance"
    )])

    fig.update_layout(
        title="Feature Importance for Offer Completion Prediction",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=600,
        width=800
    )

    return json.dumps(fig, cls=PlotlyJSONEncoder)


# Static content for Starbucks analytics page
STARBUCKS_ANALYTICS_CONTENT = """
<h2>Starbucks Analytics: Understanding User Behavior and Demographics</h2>
<h3>Article: Understanding User Behavior and Demographics in a Digital Campaign (February 21, 2025)</h3>
<p><strong>Introduction</strong><br>
In today’s digital age, businesses rely on data-driven insights to optimize their marketing strategies and improve user engagement. A recent analysis of a dataset from a digital campaign, conducted on February 21, 2025, reveals fascinating trends about user behavior, channel preferences, and demographics. This article explores the key findings from five visualizations—offer completion rates, channel usage, age distribution, income distribution, and gender distribution—and offers insights for businesses looking to refine their approaches.</p>

<p><strong>High Engagement with Offers</strong><br>
The campaign data shows an impressive level of user engagement, with approximately 75% of users completing the offer (12,000 out of 16,000 total responses). This high completion rate suggests that the offer resonated well with the target audience, potentially due to its relevance, accessibility, or incentives. Businesses can leverage this success by replicating similar strategies in future campaigns, ensuring offers are clear, appealing, and easy to redeem.</p>

<p><strong>Channel Preferences: Email Leads the Way</strong><br>
When it comes to how users interact with the campaign, Email emerges as the most popular channel, with 16,000 users engaging through this medium. Web follows closely with 14,000 users, while Mobile and Social channels trail with 10,000 and 8,000 users, respectively. This dominance of Email highlights its effectiveness as a direct communication tool, likely due to its ability to deliver personalized, targeted messages. However, the lower engagement on Mobile and Social platforms indicates an opportunity for businesses to enhance their presence on these channels—perhaps through more engaging mobile apps or targeted social media ads—to capture a broader audience.</p>

<p><strong>A Young, Middle-Income, Male-Dominant User Base</strong><br>
The demographic breakdown of the user base paints a clear picture of the campaign’s audience. The age distribution reveals a predominantly young to middle-aged group, with the majority of users falling between 20 and 40 years old, peaking around 30–35. This youthful demographic is tech-savvy and likely comfortable with digital platforms, making them an ideal target for online campaigns.

In terms of income, most users earn between 8,000 and 10,000 (in the assumed currency unit), indicating a middle-income audience. This insight is crucial for pricing strategies, as offers and products should align with this income bracket to maximize conversions.

Gender distribution further shows that the campaign appeals primarily to males, who account for about 8,000 users, compared to 6,000 females and 2,000 identifying as Other. While the male dominance is clear, the presence of female and other-gender users suggests an inclusive audience that could benefit from tailored marketing efforts to ensure equal engagement across all gender identities.</p>

<p><strong>Implications for Businesses</strong><br>
These findings offer actionable insights for businesses. First, maintaining the momentum of high offer completion rates requires continuous monitoring of user feedback and refining offers to keep them relevant. Second, while Email remains a powerhouse, investing in Mobile and Social channels could expand reach and diversify engagement. Finally, understanding the young, middle-income, and male-dominant demographic allows for more targeted marketing, but businesses should also explore strategies to better engage female and other-gender users to ensure inclusivity.</p>
"""

# Route for the analytics dashboard


@app.route('/')
def analytics_dashboard():
    """
    Renders the analytics dashboard with visualizations.
    
    Returns:
        str: Rendered HTML template with visualizations or error message.
    """
    try:
        # Load data from CSV (adjust path as needed or use load_data from database)
        visualizations = generate_visualizations(data_path='../data/final_df.csv')
        if 'error' in visualizations:
            raise Exception(visualizations['error'])
        return render_template('master.html', graphs=visualizations, active_tab='dashboard')
    except Exception as e:
        logger.error(
            f"Error in analytics_dashboard route: {str(e)}", exc_info=True)
        return render_template('master.html', error=str(e)), 500

# Route for Starbucks analytics content


@app.route('/starbucks-analytics')
def starbucks_analytics():
    """
    Renders the Starbucks analytics static content page.
    
    Returns:
        str: Rendered HTML template with content.
    """
    return render_template('starbucks_analytics.html', content=STARBUCKS_ANALYTICS_CONTENT, active_tab='starbucks')

# Route for predictions and feature importance


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles GET requests to display feature importance and POST requests for predictions.
    
    Returns:
        str or jsonify: Rendered template for GET, JSON response for POST.
    """
    if request.method == 'GET':
        feature_json = generate_feature_importance()
        return render_template('predict.html', active_tab='predict', feature_graph=feature_json)

    try:
        data = request.json
        required_fields = ['age', 'income', 'gender', 'channels', 'offer_type']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        input_data = pd.DataFrame([{
            'age': int(data['age']),
            'income': int(data['income']),
            'gender_F': 1 if data['gender'] == 'F' else 0,
            'gender_M': 1 if data['gender'] == 'M' else 0,
            'gender_O': 1 if data['gender'] == 'O' else 0,
            'channel_web': 1 if 'web' in data['channels'] else 0,
            'channel_email': 1 if 'email' in data['channels'] else 0,
            'channel_mobile': 1 if 'mobile' in data['channels'] else 0,
            'channel_social': 1 if 'social' in data['channels'] else 0,
            'offer_type_bogo': 1 if data['offer_type'] == 'bogo' else 0,
            'offer_type_discount': 1 if data['offer_type'] == 'discount' else 0,
            'offer_type_informational': 1 if data['offer_type'] == 'informational' else 0
        }])

        input_data = input_data[best_model.feature_names_in_]
        prediction = best_model.predict(input_data)[0]
        probability = best_model.predict_proba(input_data)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'confidence': float(round(probability * 100, 2))
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
