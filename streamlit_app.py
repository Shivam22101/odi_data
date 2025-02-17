import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "plotly"])
    import plotly.graph_objects as go
 

# Load the data
@st.cache

def optimize_dataframe(df):
    """Optimize data types to reduce memory usage."""
    for col in df.columns:
        col_type = df[col].dtype

        if col_type == 'object':  # Convert text columns to category
            df[col] = df[col].astype('category')

        elif col_type in ['int64', 'int32']:
            if df[col].min() >= np.iinfo(np.int8).min and df[col].max() <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif col_type in ['float64', 'float32']:
            df[col] = df[col].astype(np.float32)  # Convert to float32

    return df

def load_and_optimize_csv(filename, chunksize=100000):
    """Load CSV in chunks, optimize data types, and merge."""
    chunks = []
    
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        optimized_chunk = optimize_dataframe(chunk)
        chunks.append(optimized_chunk)

    return pd.concat(chunks, ignore_index=True)

# Load and optimize the data
data = load_and_optimize_csv('https://drive.google.com/uc?id=1RqPHrnZ8MpgfeyO45TMXMihYWYo9q1Em&export=download')

 


# Sidebar filters
st.sidebar.header("Filters")

# Multi-select filters for all possible columns
selected_batsmen = st.sidebar.multiselect("Select Batsmen", data['bat'].unique())
selected_bowlers = st.sidebar.multiselect("Select Bowlers", data['bowl'].unique())
selected_teams = st.sidebar.multiselect("Select Teams", data['team_bat'].unique())
selected_competitions = st.sidebar.multiselect("Select Competitions", data['competition'].unique())
selected_grounds = st.sidebar.multiselect("Select Grounds", data['ground'].unique())
selected_countries = st.sidebar.multiselect("Select Countries", data['country'].unique())
selected_bat_hand = st.sidebar.multiselect("Select Batting Hand", data['bat_hand'].unique())
selected_bowl_style = st.sidebar.multiselect("Select Bowling Style", data['bowl_style'].unique())
selected_bowl_kind = st.sidebar.multiselect("Select Bowling Kind", data['bowl_kind'].unique())
selected_outcome = st.sidebar.multiselect("Select Outcome", data['outcome'].unique())
selected_dismissal = st.sidebar.multiselect("Select Dismissal", data['dismissal'].unique())
selected_line = st.sidebar.multiselect("Select Line", data['line'].unique())
selected_length = st.sidebar.multiselect("Select Length", data['length'].unique())
selected_shot = st.sidebar.multiselect("Select Shot", data['shot'].unique())
selected_control = st.sidebar.multiselect("Select Control", data['control'].unique())
selected_rain = st.sidebar.multiselect("Select Rain", data['rain'].unique())
selected_daynight = st.sidebar.multiselect("Select Day/Night", data['daynight'].unique())

# Apply filters
filtered_data = data.copy(deep=False)  # Creates a shallow copy, saving memory

if selected_batsmen:
    filtered_data = filtered_data[filtered_data['bat'].isin(selected_batsmen)]
if selected_bowlers:
    filtered_data = filtered_data[filtered_data['bowl'].isin(selected_bowlers)]
if selected_teams:
    filtered_data = filtered_data[filtered_data['team_bat'].isin(selected_teams)]
if selected_competitions:
    filtered_data = filtered_data[filtered_data['competition'].isin(selected_competitions)]
if selected_grounds:
    filtered_data = filtered_data[filtered_data['ground'].isin(selected_grounds)]
if selected_countries:
    filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]
if selected_bat_hand:
    filtered_data = filtered_data[filtered_data['bat_hand'].isin(selected_bat_hand)]
if selected_bowl_style:
    filtered_data = filtered_data[filtered_data['bowl_style'].isin(selected_bowl_style)]
if selected_bowl_kind:
    filtered_data = filtered_data[filtered_data['bowl_kind'].isin(selected_bowl_kind)]
if selected_outcome:
    filtered_data = filtered_data[filtered_data['outcome'].isin(selected_outcome)]
if selected_dismissal:
    filtered_data = filtered_data[filtered_data['dismissal'].isin(selected_dismissal)]
if selected_line:
    filtered_data = filtered_data[filtered_data['line'].isin(selected_line)]
if selected_length:
    filtered_data = filtered_data[filtered_data['length'].isin(selected_length)]
if selected_shot:
    filtered_data = filtered_data[filtered_data['shot'].isin(selected_shot)]
if selected_control:
    filtered_data = filtered_data[filtered_data['control'].isin(selected_control)]
if selected_rain:
    filtered_data = filtered_data[filtered_data['rain'].isin(selected_rain)]
if selected_daynight:
    filtered_data = filtered_data[filtered_data['daynight'].isin(selected_daynight)]

# Display filtered data (only the first 100 rows)
st.write("Filtered Data (First 100 Rows)")
st.write(filtered_data.head(100))

# Check if filtered data is empty
if filtered_data.empty:
    st.write("No data available for the selected filters.")
else:
    # Calculate metrics
    total_runs = filtered_data['batruns'].sum()
    total_balls = filtered_data['ballfaced'].sum()

    # Calculate total wickets (dismissals)
    total_wickets = filtered_data[filtered_data['outcome'] == 'out'].shape[0]

    runs_per_ball = total_runs / total_balls if total_balls > 0 else 0
    balls_per_wicket = total_balls / total_wickets if total_wickets > 0 else 0
    runs_per_wicket = total_runs / total_wickets if total_wickets > 0 else 0

    st.write("### Metrics")
    st.write(f"Total Runs: {total_runs}")
    st.write(f"Total Balls: {total_balls}")
    st.write(f"Total Wickets: {total_wickets}")
    st.write(f"Runs per Ball: {runs_per_ball:.2f}")
    st.write(f"Balls per Wicket: {balls_per_wicket:.2f}")
    st.write(f"Runs per Wicket: {runs_per_wicket:.2f}")

    # Handle missing or invalid values
    filtered_data['line'].fillna('Unknown', inplace=True)
    filtered_data['length'].fillna('Unknown', inplace=True)
    filtered_data['ballfaced'].replace(0, np.nan, inplace=True)

    # Calculate Strike Rate (SR)
    filtered_data['strike_rate'] = (filtered_data['batruns'] / filtered_data['ballfaced']) * 100

    # Calculate Runs per Wicket
    # Create a column to mark wickets (dismissals)
    filtered_data['is_wicket'] = filtered_data['outcome'] == 'out'
    total_runs_by_line_length = filtered_data.groupby(['line', 'length'])['batruns'].sum()
    total_wickets_by_line_length = filtered_data.groupby(['line', 'length'])['is_wicket'].sum()
    runs_per_wicket_by_line_length = (total_runs_by_line_length / total_wickets_by_line_length.replace(0, np.nan)).reset_index()
    runs_per_wicket_by_line_length.rename(columns={0: 'runs_per_wicket'}, inplace=True)

    # Merge runs_per_wicket back into the original DataFrame
    filtered_data = filtered_data.merge(
        runs_per_wicket_by_line_length,
        on=['line', 'length'],
        how='left'
    )

    # Pivot table: Line-Length Combo with Average Strike Rate and Runs per Wicket
    pivot_table_sr = pd.pivot_table(
        filtered_data,
        values='strike_rate',
        index='line',
        columns='length',
        aggfunc=np.mean,
        fill_value=0
    )

    pivot_table_rpw = pd.pivot_table(
        filtered_data,
        values='runs_per_wicket',
        index='line',
        columns='length',
        aggfunc=np.mean,
        fill_value=0
    )

    st.write("### Line-Length Combo Pivot Table (Average Strike Rate)")
    st.write(pivot_table_sr)

    st.write("### Line-Length Combo Pivot Table (Runs per Wicket)")
    st.write(pivot_table_rpw)

    # Pivot table: Average SR for Pace vs Spin
    pivot_table_pace_spin = pd.pivot_table(
        filtered_data,
        values=['strike_rate','runs_per_wicket'],
        index='bowl_kind',
        aggfunc=np.mean,
        fill_value=0
    )
    st.write("### Average Strike Rate: Pace vs Spin")
    st.write(pivot_table_pace_spin)

    # Pivot table: Average SR for Various Bowling Types
    pivot_table_bowl_style = pd.pivot_table(
        filtered_data,
        values=['strike_rate','runs_per_wicket'],
        index='bowl_style',
        aggfunc=np.mean,
        fill_value=0
    )
    st.write("### Average Strike Rate: Bowling Styles")
    st.write(pivot_table_bowl_style)

 
