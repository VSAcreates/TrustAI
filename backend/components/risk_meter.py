import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def render_risk_meter():
    """
    Render the risk meter gauge and threshold controls
    """
    st.subheader("Risk Meter")
    st.caption("Current fraud risk level")
    
    # Create risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.current_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': 
                   "#ef4444" if st.session_state.current_risk > 80 else 
                   "#f97316" if st.session_state.current_risk > 50 else 
                   "#22c55e"},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [50, 80], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': st.session_state.risk_threshold
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk threshold slider
    st.caption(f"Risk Threshold: {st.session_state.risk_threshold}%")
    risk_threshold = st.slider("", 0, 100, st.session_state.risk_threshold)
    if risk_threshold != st.session_state.risk_threshold:
        st.session_state.risk_threshold = risk_threshold

def render_risk_breakdown():
    """
    Render risk breakdown charts
    """
    st.subheader("Risk Breakdown")
    
    # Sample data for risk breakdown
    risk_categories = {
        "Location": st.session_state.current_risk * 0.3,
        "Amount": st.session_state.current_risk * 0.25,
        "Frequency": st.session_state.current_risk * 0.2,
        "Device": st.session_state.current_risk * 0.15,
        "Time": st.session_state.current_risk * 0.1
    }
    
    # Create a DataFrame for the breakdown
    df = pd.DataFrame({
        'Category': list(risk_categories.keys()),
        'Risk': list(risk_categories.values())
    })
    
    # Sort by risk level
    df = df.sort_values('Risk', ascending=False)
    
    # Create horizontal bar chart
    fig = px.bar(
        df, 
        x='Risk', 
        y='Category', 
        orientation='h',
        color='Risk',
        color_continuous_scale=['green', 'yellow', 'orange', 'red'],
        range_color=[0, 100]
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Risk Score",
        yaxis_title="",
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_risk_trends():
    """
    Render risk trend over time
    """
    st.subheader("Risk Trend")
    
    # Generate sample risk trend data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=7, freq='D')
    base_risk = st.session_state.current_risk
    
    # Create some variation in the risk level
    risk_levels = [
        max(0, min(100, base_risk + np.random.normal(0, 10))) 
        for _ in range(len(dates))
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Risk': risk_levels
    })
    
    # Create line chart
    fig = px.line(
        df,
        x='Date',
        y='Risk',
        markers=True,
        line_shape='spline'
    )
    
    # Add threshold line
    fig.add_hline(
        y=st.session_state.risk_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def update_risk_score(new_transaction_risk):
    """
    Update the current risk score based on new transaction
    
    Args:
        new_transaction_risk: Risk score of the new transaction
    """
    # Update current risk (moving average)
    st.session_state.current_risk = round((st.session_state.current_risk * 2 + new_transaction_risk) / 3)
    return st.session_state.current_risk