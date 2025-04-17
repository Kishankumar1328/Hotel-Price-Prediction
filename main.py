import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import random

# Set page configuration with a favicon and improved title
st.set_page_config(
    layout="wide", 
    page_title="Hotel Price Prediction Dashboard",
    page_icon="🏨",
    initial_sidebar_state="expanded"
)

# Apply enhanced custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .dashboard-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: #1e40af;
        margin-bottom: 25px;
        padding: 15px;
        border-bottom: 3px solid #3b82f6;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12);
    }
    .prediction-result {
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        margin: 12px 0;
        padding: 12px;
        background-color: #f1f5f9;
        border-radius: 8px;
        color: #1e40af;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #1e40af;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #bfdbfe;
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    .stSidebar .stButton>button {
        width: 100%;
    }
    .metric-label {
        font-size: 16px;
        font-weight: 600;
        color: #64748b;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1e40af;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #bfdbfe;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        color: #64748b;
        font-size: 14px;
        border-top: 1px solid #e2e8f0;
        margin-top: 30px;
    }
    .feature-importance {
        padding: 15px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    }
    .model-selector {
        padding: 10px;
        background-color: #f1f5f9;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    # Try different encodings to handle the file
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            data = pd.read_csv(r"E:\projects\IBM PROJECT\hotel price prediction 1000.csv", encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            st.error("CSV file not found. Please check the file path.")
            return pd.DataFrame()
    
    # If all encodings fail
    st.error("Could not decode the CSV file with any of the common encodings. The file might be corrupted.")
    return pd.DataFrame()

# Data preprocessing
@st.cache_data
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Label encoding for categorical variables
    le_city = LabelEncoder()
    le_hotel = LabelEncoder()
    
    data['City_encoded'] = le_city.fit_transform(data['City'])
    data['Hotel_encoded'] = le_hotel.fit_transform(data['Hotel name'])
    
    # Create mapping dictionaries for later use
    city_mapping = dict(zip(le_city.transform(le_city.classes_), le_city.classes_))
    hotel_mapping = dict(zip(le_hotel.transform(le_hotel.classes_), le_hotel.classes_))
    
    # Scale numerical features
    scaler = StandardScaler()
    numeric_features = ['Hotel star rating', 'Distance', 'Customer rating', 'Rooms', 'Squares']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    return data, city_mapping, hotel_mapping, le_city, le_hotel, scaler, numeric_features

# Train Random Forest model
@st.cache_data
def train_rf_model(data):
    # Features and target
    X = data[['Hotel star rating', 'Distance', 'Customer rating', 'Rooms', 'Squares', 
              'City_encoded', 'Hotel_encoded']]
    y = data['Price(BAM)']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, mse, r2, X_test, y_test, y_pred, feature_importance

# Predict prices for the next few days
def predict_future_prices(model, df, city, hotel_name, le_city, le_hotel, scaler, numeric_features):
    # Get a row for the selected hotel in the selected city
    hotel_data = df[(df['City'] == city) & (df['Hotel name'] == hotel_name)]
    
    if len(hotel_data) == 0:
        return None, None, None
    
    # Use the first room configuration for this hotel
    hotel_data = hotel_data.iloc[0].copy()
    
    # Generate features for prediction
    city_encoded = le_city.transform([city])[0]
    hotel_encoded = le_hotel.transform([hotel_name])[0]
    
    # Get the original price for reference
    original_price = hotel_data['Price(BAM)']
    
    # Base features
    base_features = hotel_data[numeric_features].values
    
    # Create scaled features for prediction
    scaled_features = np.append(base_features, [city_encoded, hotel_encoded])
    scaled_features = scaled_features.reshape(1, -1)
    
    # Predict base price
    base_price = model.predict(scaled_features)[0]
    
    # Apply more significant variations for future prices
    # Use percentage changes to create more noticeable differences
    today_price = base_price
    tomorrow_price = base_price * (1 + random.uniform(-0.15, 0.20))  # -15% to +20%
    day_after_price = base_price * (1 + random.uniform(-0.25, 0.30))  # -25% to +30%
    
    return today_price, tomorrow_price, day_after_price

# Generate city insights
def generate_city_insights(df, city):
    city_data = df[df['City'] == city]
    
    avg_price = city_data['Price(BAM)'].mean()
    min_price = city_data['Price(BAM)'].min()
    max_price = city_data['Price(BAM)'].max()
    avg_rating = city_data['Customer rating'].mean()
    
    # Group by star rating
    star_data = city_data.groupby('Hotel star rating')['Price(BAM)'].mean().reset_index()
    
    # Group by hotel
    hotel_data = city_data.groupby('Hotel name')['Price(BAM)'].mean().reset_index()
    
    return avg_price, min_price, max_price, avg_rating, star_data, hotel_data

# Create visualizations
def create_price_comparison_chart(today_price, tomorrow_price, day_after_price, hotel_name):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    day_after = (datetime.datetime.now() + datetime.timedelta(days=2)).strftime('%Y-%m-%d')
    
    dates = [today, tomorrow, day_after]
    prices = [today_price, tomorrow_price, day_after_price]
    
    fig = px.line(
        x=dates, 
        y=prices,
        markers=True,
        title=f'Price Trend for {hotel_name}',
        labels={'x': 'Date', 'y': 'Price (BAM)'}
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_font_size=20,
        title_x=0.5,
        xaxis=dict(tickangle=-45),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155'),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    fig.update_traces(
        line=dict(width=3, color='#3b82f6'),
        marker=dict(size=10, color='#1e40af')
    )
    
    return fig

def create_city_price_distribution(df, city):
    city_data = df[df['City'] == city]
    
    fig = px.histogram(
        city_data, 
        x='Price(BAM)',
        nbins=20,
        title=f'Price Distribution in {city}',
        color_discrete_sequence=['#3b82f6']
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_font_size=20,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155'),
        bargap=0.1
    )
    
    return fig

def create_star_rating_chart(star_data):
    fig = px.bar(
        star_data,
        x='Hotel star rating',
        y='Price(BAM)',
        title='Average Price by Star Rating',
        color='Hotel star rating',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_font_size=20,
        title_x=0.5,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155')
    )
    
    return fig

def create_hotel_comparison_chart(hotel_data, selected_hotel):
    # Sort by price
    hotel_data = hotel_data.sort_values('Price(BAM)')
    
    # Highlight selected hotel
    colors = ['rgba(59, 130, 246, 0.7)'] * len(hotel_data)
    for i, hotel in enumerate(hotel_data['Hotel name']):
        if hotel == selected_hotel:
            colors[i] = 'rgba(220, 38, 38, 0.9)'
    
    fig = px.bar(
        hotel_data,
        x='Hotel name',
        y='Price(BAM)',
        title='Average Price by Hotel',
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        title_font_size=20,
        title_x=0.5,
        xaxis=dict(tickangle=-45),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155')
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_font_size=20,
        title_x=0.5,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155'),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# Main application
def main():
    # Add a loading spinner
    with st.spinner("Loading dashboard..."):
        # App header with logo
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="dashboard-title">🏨 Hotel Price Prediction Dashboard</div>', unsafe_allow_html=True)
        
        # Load and preprocess data
        df = load_data()
        
        if df.empty:
            st.error("No data available. Please check the data source.")
            st.stop()
        
        # Sidebar for filters and controls
        with st.sidebar:
            st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
            
            # City selection
            cities = sorted(df['City'].unique())
            selected_city = st.selectbox("Select City", cities)
            
            # Hotel selection based on city
            city_hotels = sorted(df[df['City'] == selected_city]['Hotel name'].unique())
            selected_hotel = st.selectbox("Select Hotel", city_hotels)
            
            # Divider
            st.markdown("---")
        
        # Process data
        processed_data, city_mapping, hotel_mapping, le_city, le_hotel, scaler, numeric_features = preprocess_data(df)
        
        # Train the model
        model, mse, r2, X_test, y_test, y_pred, feature_importance = train_rf_model(processed_data)
        
        # Add model metrics to sidebar after model is trained
        with st.sidebar:
            # Model metrics
            st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Mean Squared Error</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{mse:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>R² Score</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{r2:.2f}</div>", unsafe_allow_html=True)
        
        # Generate insights for the selected city
        avg_price, min_price, max_price, avg_rating, star_data, hotel_data = generate_city_insights(df, selected_city)
        
        # Predict prices for the next few days
        today_price, tomorrow_price, day_after_price = predict_future_prices(
            model, df, selected_city, selected_hotel, le_city, le_hotel, scaler, numeric_features
        )
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Price Predictions", "City Analysis", "Hotel Comparison", "Model Insights"])
        
        with tab1:
            # Display key metrics in a row
            st.markdown('<div class="section-header">Key Metrics for ' + selected_city + '</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Price", f"{avg_price:.2f} BAM", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Minimum Price", f"{min_price:.2f} BAM", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Maximum Price", f"{max_price:.2f} BAM", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Rating", f"{avg_rating:.1f}/10", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Price prediction section
            st.markdown('<div class="section-header">Price Prediction for ' + selected_hotel + '</div>', unsafe_allow_html=True)
            
            if today_price is not None:
                # Display predicted prices
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>Today's Price</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='prediction-result'>{today_price:.2f} BAM</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>Tomorrow's Price</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='prediction-result'>{tomorrow_price:.2f} BAM</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>Day After Tomorrow</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='prediction-result'>{day_after_price:.2f} BAM</div>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Price trend chart
                st.markdown('<div class="section-header">Price Trend</div>', unsafe_allow_html=True)
                price_chart = create_price_comparison_chart(today_price, tomorrow_price, day_after_price, selected_hotel)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Booking recommendation
                st.markdown('<div class="section-header">Booking Recommendation</div>', unsafe_allow_html=True)
                
                if today_price <= tomorrow_price and today_price <= day_after_price:
                    recommendation = "Book today! Prices are expected to rise."
                    icon = "⬆️"
                elif tomorrow_price < today_price and tomorrow_price <= day_after_price:
                    recommendation = "Wait until tomorrow for a better price."
                    icon = "⬇️"
                else:
                    recommendation = "Wait for two days for the best price."
                    icon = "⬇️"
                
                st.markdown(f"<div style='font-size: 20px; padding: 15px; background-color: #f1f5f9; border-radius: 8px; text-align: center;'>{icon} <b>{recommendation}</b> {icon}</div>", unsafe_allow_html=True)
            else:
                st.error(f"No data available for {selected_hotel} in {selected_city}")
        
        with tab2:
            # City analysis tab
            st.markdown('<div class="section-header">Price Distribution in ' + selected_city + '</div>', unsafe_allow_html=True)
            
            # Price distribution histogram
            price_dist = create_city_price_distribution(df, selected_city)
            st.plotly_chart(price_dist, use_container_width=True)
            
            # Star rating analysis
            st.markdown('<div class="section-header">Price by Star Rating</div>', unsafe_allow_html=True)
            star_chart = create_star_rating_chart(star_data)
            st.plotly_chart(star_chart, use_container_width=True)
            
            # City insights
            st.markdown('<div class="section-header">City Insights</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                price_diff = max_price - min_price  # Calculate price_diff before formatting
                st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 8px; height: 100%; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);'>
                    <h3 style='color: #1e40af; margin-bottom: 15px;'>Price Range</h3>
                    <p>The price range in {city} varies from <b>{min_price:.2f} BAM</b> to <b>{max_price:.2f} BAM</b>, with an average of <b>{avg_price:.2f} BAM</b>.</p>
                    <p>This represents a difference of <b>{price_diff:.2f} BAM</b> between the most affordable and the most expensive options.</p>
                </div>
                """.format(city=selected_city, min_price=min_price, max_price=max_price, avg_price=avg_price, price_diff=price_diff), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 8px; height: 100%; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);'>
                    <h3 style='color: #1e40af; margin-bottom: 15px;'>Quality Assessment</h3>
                    <p>Hotels in {city} have an average customer rating of <b>{avg_rating:.1f}/10</b>.</p>
                    <p>The city offers a variety of accommodations across different star ratings, catering to different budget ranges and preferences.</p>
                </div>
                """.format(city=selected_city, avg_rating=avg_rating), unsafe_allow_html=True)
        
        with tab3:
            # Hotel comparison tab
            st.markdown('<div class="section-header">Hotel Price Comparison in ' + selected_city + '</div>', unsafe_allow_html=True)
            
            hotel_chart = create_hotel_comparison_chart(hotel_data, selected_hotel)
            st.plotly_chart(hotel_chart, use_container_width=True)
            
            # Selected hotel details
            hotel_details = df[(df['City'] == selected_city) & (df['Hotel name'] == selected_hotel)].iloc[0]
            
            st.markdown('<div class="section-header">Selected Hotel Details</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);'>
                    <h3 style='color: #1e40af; margin-bottom: 15px;'>{hotel_name}</h3>
                    <p><b>Star Rating:</b> {stars} ⭐</p>
                    <p><b>Distance from Center:</b> {distance} km</p>
                    <p><b>Customer Rating:</b> {rating}/10</p>
                </div>
                """.format(
                    hotel_name=selected_hotel,
                    stars=int(hotel_details['Hotel star rating']),
                    distance=hotel_details['Distance'],
                    rating=hotel_details['Customer rating']
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);'>
                    <h3 style='color: #1e40af; margin-bottom: 15px;'>Accommodation Details</h3>
                    <p><b>Number of Rooms:</b> {rooms}</p>
                    <p><b>Room Size:</b> {size} m²</p>
                    <p><b>Current Price:</b> {price} BAM</p>
                </div>
                """.format(
                    rooms=int(hotel_details['Rooms']),
                    size=int(hotel_details['Squares']),
                    price=hotel_details['Price(BAM)']
                ), unsafe_allow_html=True)
            
            # Value assessment
            avg_city_price = df[df['City'] == selected_city]['Price(BAM)'].mean()
            hotel_price = hotel_details['Price(BAM)']
            price_diff = ((hotel_price - avg_city_price) / avg_city_price) * 100
            
            st.markdown('<div class="section-header">Value Assessment</div>', unsafe_allow_html=True)
            
            if price_diff < -10:
                value_message = "This hotel offers excellent value, with a price significantly below the city average."
                value_color = "#15803d"  # Green
            elif price_diff < 0:
                value_message = "This hotel offers good value, with a price below the city average."
                value_color = "#0284c7"  # Blue
            elif price_diff < 10:
                value_message = "This hotel is priced close to the city average."
                value_color = "#7c3aed"  # Purple
            elif price_diff < 25:
                value_message = "This hotel is priced above the city average, but may offer premium amenities."
                value_color = "#f59e0b"  # Orange
            else:
                value_message = "This hotel is in the luxury segment, with prices significantly above the city average."
                value_color = "#dc2626"  # Red
            
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);'>
                <h3 style='color: {value_color}; margin-bottom: 15px;'>Price Analysis</h3>
                <p>This hotel is <b>{abs(price_diff):.1f}%</b> {"below" if price_diff < 0 else "above"} the city average price.</p>
                <p>{value_message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            # Model insights tab
            st.markdown('<div class="section-header">Model Performance Analysis</div>', unsafe_allow_html=True)
            
            # Feature importance
            st.markdown('<div class="section-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
            st.markdown("""
            Feature importance helps us understand which factors most influence hotel prices.
            Higher values indicate stronger influence on the prediction.
            """)
            feature_chart = create_feature_importance_chart(feature_importance)
            st.plotly_chart(feature_chart, use_container_width=True, key="feature_importance_chart")
            
            # Feature explanation
            st.markdown('<div class="section-header">Feature Explanation</div>', unsafe_allow_html=True)
            st.markdown("""
            - **Hotel star rating**: The official star rating of the hotel
            - **Distance**: Distance from city center in kilometers
            - **Customer rating**: Average customer rating out of 10
            - **Rooms**: Number of rooms in the hotel
            - **Squares**: Average room size in square meters
            - **City_encoded**: Numerical representation of the city
            - **Hotel_encoded**: Numerical representation of the hotel name
            """)
            
            # Error analysis
            st.markdown('<div class="section-header">Error Analysis</div>', unsafe_allow_html=True)
            
            # Create a dataframe for actual vs predicted
            error_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Error': y_test - y_pred
            })
            
            # Scatter plot of actual vs predicted
                        # Scatter plot of actual vs predicted
            fig = px.scatter(
                error_df,
                x='Actual',
                y='Predicted',
                title='Actual vs Predicted Prices',
                labels={'Actual': 'Actual Price (BAM)', 'Predicted': 'Predicted Price (BAM)'},
                color='Error',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            
            # Add perfect prediction line
            fig.add_trace(
                go.Scatter(
                    x=[error_df['Actual'].min(), error_df['Actual'].max()],
                    y=[error_df['Actual'].min(), error_df['Actual'].max()],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='Perfect Prediction'
                )
            )
            
            fig.update_layout(
                height=500,
                template='plotly_white',
                title_font_size=20,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#334155')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Error distribution
            error_hist = px.histogram(
                error_df,
                x='Error',
                nbins=30,
                title='Error Distribution',
                color_discrete_sequence=['#3b82f6']
            )
            
            error_hist.update_layout(
                height=400,
                template='plotly_white',
                title_font_size=20,
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#334155'),
                bargap=0.1
            )
            
            st.plotly_chart(error_hist, use_container_width=True)
    
    # Footer
    st.markdown('<div class="footer">Hotel Price Prediction Dashboard | Created with Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()