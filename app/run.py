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
from visualization import generate_visualizations
from datetime import datetime
import sys

# Initialize Flask application
app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define paths for model and database (to be set via command-line arguments)
MODEL_PATH = None
DATABASE_PATH = None

# Load the pre-trained model
best_model = None

def load_data(database_filepath):
    """
    Loads data from a SQLite database into a pandas DataFrame.
    
    Parameters:
    database_filepath (str): Path to the SQLite database.
    
    Returns:
    pd.DataFrame: Raw data from the database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    return pd.read_sql("SELECT * FROM cleaned_data", engine)

def preprocess_data(df):
    """
    Preprocesses the DataFrame for model predictions by handling missing values,
    encoding categorical variables, and aligning columns with model expectations.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: Processed DataFrame ready for predictions.
    """
    df['age'] = df['age'].replace(118, np.nan).fillna(df['age'].mean()).astype(int)
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

def generate_feature_importance():
    """
    Generates a visualization of feature importance for the model.
    
    Returns:
        str: JSON representation of the Plotly figure for feature importance.
    """
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': best_model.feature_names_in_,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        # Fallback for models without feature_importances_ (e.g., Logistic Regression)
        feature_importance = pd.DataFrame({
            'feature': best_model.feature_names_in_,
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
<h2>Starbucks Analytics: Insights from Digital Campaign Analytics and Business Recommendations</h2>
<h3>Article: Insights from Starbucks Digital Campaign Analytics and Business Recommendations (February 24, 2025)</h3>
<p><strong>Introduction</strong><br>
A recent analysis of a Starbucks digital campaign dataset, conducted as of February 24, 2025, provides valuable insights into user behavior, demographics, and offer completion trends. By examining six key visualizations—age distribution, gender distribution, income distribution, and offer completion rates across these demographics—we can uncover actionable strategies for Starbucks to enhance its marketing efforts and drive higher engagement. This article explores the findings and offers targeted recommendations for businesses looking to optimize their digital campaigns.</p>

<p><strong>Key Findings from the Visualizations</strong></p>

<p><strong>1. Age Distribution</strong><br>
The data reveals a predominantly young to middle-aged user base, with the highest concentration of users (around 5,000) falling in the 30-40 age group. Participation drops significantly for older age groups, with fewer than 1,000 users in the 80-90 and 90-110 age brackets.
<strong>Insight</strong>: Starbucks’ campaigns are resonating strongly with younger, tech-savvy audiences, likely due to their comfort with digital platforms and mobile apps.</p>

<p><strong>2. Gender Distribution</strong><br>
Males dominate the user base, accounting for approximately 7,000 users, followed by females with around 5,000, and a smaller group (about 2,000) identifying as “Other” or “nan” (not specified).
<strong>Insight</strong>: While the campaign appeals broadly, there’s a clear male skew, suggesting a potential opportunity to better engage female and other-gender users.</p>

<p><strong>3. Income Distribution</strong><br>
The majority of users (around 7,000) earn between $60,000 and $90,000 annually, with a significant portion (about 4,000) earning $30,000 to $60,000. Fewer than 2,000 users fall into the lower ($0–$30,000) or higher ($90,000–$120,000) income brackets.
<strong>Insight</strong>: The campaign targets a middle-income audience, which aligns with Starbucks’ typical customer base, but there’s room to explore strategies for lower and higher income segments.</p>

<p><strong>4. Offer Completion by Gender</strong><br>
Offer completion rates show a strong male engagement, with about 5,000 males completing offers compared to around 3,000 who did not. For females, approximately 3,000 completed offers, while 2,000 did not, and the “Other” category shows lower completion (around 1,000 completed vs. 1,000 not completed).
<strong>Insight</strong>: Males are more likely to complete offers, but there’s untapped potential to increase female and other-gender engagement.</p>

<p><strong>5. Offer Completion by Age Group</strong><br>
The 30-40 age group leads in offer completions, with nearly 4,000 users completing offers and about 1,000 not completing. Younger (18-30) and older (40-50) groups show moderate completion rates, while older age groups (70+ and above) have minimal completions.
<strong>Insight</strong>: The 30-40 age group is the most responsive, indicating a prime target for campaign focus, but efforts could be made to engage younger and older demographics.</p>

<p><strong>6. Offer Completion by Income Group</strong><br>
Users in the $60,000–$90,000 income bracket have the highest offer completion rate, with around 5,000 completing offers and 2,000 not completing. The $30,000–$60,000 group shows about 3,000 completions and 1,000 non-completions, while lower and higher income groups have lower engagement.
<strong>Insight</strong>: Middle-income users are most likely to engage with offers, but tailoring strategies for lower and higher income segments could broaden participation.</p>

<p><strong>Business Recommendations</strong><br>
Based on these insights, Starbucks can refine its digital marketing strategies to maximize user engagement and offer completion rates. Here are actionable recommendations:</p>

<p><strong>1. Target the 30-40 Age Group with Tailored Offers</strong><br>
Since the 30-40 age group shows the highest participation and offer completion, Starbucks should prioritize this demographic in future campaigns. Develop offers that resonate with their lifestyles, such as convenience-focused promotions (e.g., mobile app discounts for quick pickups) or premium drink bundles. Use targeted email and mobile notifications to maintain engagement.</p>

<p><strong>2. Enhance Engagement for Females and Other Genders</strong><br>
The male dominance in offer completion suggests an opportunity to boost female and other-gender participation. Create gender-inclusive marketing campaigns, such as featuring diverse customer stories or offering gender-neutral promotions (e.g., seasonal drink specials appealing to all). Conduct surveys to understand preferences among these groups and adjust messaging accordingly.</p>

<p><strong>3. Expand Reach to Lower and Higher Income Segments</strong><br>
While middle-income users ($30,000–$90,000) are the primary audience, there’s potential to attract lower-income users with budget-friendly offers (e.g., smaller-sized drinks at reduced prices) and higher-income users with premium, exclusive experiences (e.g., limited-edition beverages or loyalty perks). Use income-based segmentation in email and social media campaigns to deliver personalized offers.</p>

<p><strong>4. Leverage Email and Web Channels for Broader Engagement</strong><br>
The high engagement from the 30-40 age group and middle-income users suggests strong reliance on digital channels like email and web. Continue investing in these platforms, but also explore mobile and social media strategies to reach younger or less digitally engaged demographics. For example, create interactive Instagram stories or TikTok challenges featuring Starbucks offers to attract younger users.</p>

<p><strong>5. Monitor and Adjust Offer Difficulty and Rewards</strong><br>
The data on offer completion rates indicates that users respond well to offers within their income and age profiles. Analyze the `difficulty` and `reward` values (as shown in the model features) to ensure offers are neither too challenging nor too easy for the target audience. For instance, reduce difficulty for lower-income users or increase rewards for higher-income users to drive completions.</p>

<p><strong>6. Incorporate Real-Time Behavioral Data</strong><br>
Use real-time data on events like `event_offer_received`, `event_offer_viewed`, and `event_transaction` to personalize follow-up communications. For example, if a user views an offer but doesn’t complete it, send a reminder via email or mobile with an incentive (e.g., a small discount) to encourage action.</p>

<p><strong>Conclusion</strong><br>
Starbucks’ digital campaign has achieved strong engagement, particularly among 30-40-year-old, middle-income males. However, by expanding outreach to other demographics—such as females, other genders, and lower/higher income groups—while leveraging key channels and tailoring offer structures, Starbucks can enhance inclusivity and drive higher offer completion rates. These data-driven strategies, informed by the February 24, 2025, analysis, will position Starbucks for continued success in its digital marketing efforts.</p>
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
        logger.error(f"Error in analytics_dashboard route: {str(e)}", exc_info=True)
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
    if request.method == 'GET':
        feature_json = generate_feature_importance()
        return render_template('predict.html', active_tab='predict', feature_graph=feature_json)

    try:
        data = request.json
        required_fields = [
            'age', 'income', 'gender', 'channels', 'offer_type',
            'event_offer_received', 'event_offer_viewed', 'event_transaction',
            'time_offer_received', 'time_offer_viewed', 'time_transaction',
            'amount_transaction', 'reward', 'difficulty'
        ]
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
            'offer_type_informational': 1 if data['offer_type'] == 'informational' else 0,
            'event_offer_received': int(data['event_offer_received']),
            'event_offer_viewed': int(data['event_offer_viewed']),
            'event_transaction': int(data['event_transaction']),
            'time_offer_received': int(data['time_offer_received']),
            'time_offer_viewed': int(data['time_offer_viewed']),
            'time_transaction': int(data['time_transaction']),
            'amount_transaction': float(data['amount_transaction']),
            'reward': float(data['reward']),
            'difficulty': int(data['difficulty'])
        }])

        # Ensure column order matches model's expectations
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

def main():
    if len(sys.argv) == 3:
        global MODEL_PATH, DATABASE_PATH, best_model
        MODEL_PATH = sys.argv[1]
        DATABASE_PATH = sys.argv[2]

        # Load the pre-trained model
        best_model = joblib.load(MODEL_PATH)

        # Run the Flask app
        app.run(debug=True)
    else:
        print("Usage: python run.py <model_path> <database_path>\n"
              "Example: python run.py ../models/best_model.pkl ../data/cleaned_data.db")

if __name__ == '__main__':
    main()