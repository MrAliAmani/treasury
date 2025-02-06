import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

class FinancialVisualizer:
    """Class to create interactive financial visualizations using Plotly"""
    
    def __init__(self):
        self.indicator_colors = {
            "Inflation": "red",
            "Growth": "blue", 
            "Unemployment": "green"
        }

    def create_surprise_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of value changes"""
        try:
            if df is None or df.empty or 'value' not in df.columns:
                return None

            fig = go.Figure()
            
            for indicator in df['indicator'].unique():
                data = df[df['indicator'] == indicator].copy()
                data['value_change'] = data['value'].diff()
                
                if len(data) < 2:
                    continue
                    
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['value_change'],
                    name=indicator,
                    mode='markers+lines'
                ))

            fig.update_layout(
                title='Value Changes Over Time',
                xaxis_title='Date',
                yaxis_title='Change in Value',
                height=500
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error in scatter plot: {e}")
            return None

    def create_yield_curve_animation(self, df: pd.DataFrame) -> go.Figure:
        """Create yield curve animation using real data"""
        try:
            # Pivot data to get values by date and indicator
            pivot_df = df.pivot(index='date', columns='indicator', values='value')
            dates = pivot_df.index.unique()
            
            fig = go.Figure()
            
            # Add initial curve
            first_date = dates[0]
            first_data = pivot_df.loc[first_date]
            
            fig.add_trace(go.Scatter(
                x=first_data.index,
                y=first_data.values,
                mode='lines+markers',
                name=str(first_date)
            ))
            
            # Create frames for animation
            frames = []
            for date in dates:
                data = pivot_df.loc[date]
                frames.append(
                    go.Frame(
                        data=[go.Scatter(
                            x=data.index,
                            y=data.values,
                            mode='lines+markers'
                        )],
                        name=str(date)
                    )
                )
            
            fig.frames = frames
            fig.update_layout(
                title="Economic Indicators Over Time",
                xaxis_title="Indicator",
                yaxis_title="Value"
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error creating yield curve animation: {str(e)}")
            return go.Figure()

    def create_impact_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap of values over time"""
        try:
            if df is None or df.empty or 'value' not in df.columns:
                return None
                
            pivot_df = df.pivot_table(
                values='value',
                index=pd.to_datetime(df['date']).dt.strftime('%Y-%m'),
                columns='indicator',
                aggfunc='mean'
            ).fillna(0)
            
            if pivot_df.empty:
                return None
                
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale='RdBu'
            ))

            fig.update_layout(
                title="Values Heatmap",
                xaxis_title="Indicator",
                yaxis_title="Date",
                height=500
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error in heatmap: {e}")
            return None

    def create_regression_dashboard(self, 
                                  regression_results: List[Dict]) -> go.Figure:
        """
        Create regression results dashboard
        
        Args:
            regression_results: List of dicts with keys ['indicator', 'slope', 'intercept']
            
        Returns:
            Plotly figure object
        """
        indicators = [r['indicator'] for r in regression_results]
        slopes = [r['slope'] for r in regression_results]
        intercepts = [r['intercept'] for r in regression_results]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Indicator</b>', '<b>Slope</b>', '<b>Intercept</b>'],
                align='center',
                line=dict(width=1, color='black'),
                fill=dict(color='lightgrey')
            ),
            cells=dict(
                values=[indicators, slopes, intercepts],
                align='center',
                line=dict(color='black', width=1)
            )
        )])

        fig.update_layout(
            title="Regression Results Dashboard",
            margin=dict(l=30, r=30, t=50, b=30)
        )
        
        return fig 

    def create_time_series(self, df: pd.DataFrame) -> go.Figure:
        """Create time series plot of actual vs expected values"""
        try:
            if df is None or df.empty or 'value' not in df.columns:
                logging.warning("Missing required data for time series plot")
                return None

            fig = go.Figure()
            
            for indicator in df['indicator'].unique():
                data = df[df['indicator'] == indicator].copy()
                if data['value'].isnull().all():
                    continue
                    
                # Plot actual values
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['value'],
                    name=f"{indicator}",
                    mode='lines+markers',
                    line=dict(color=self.indicator_colors.get(indicator, 'gray'))
                ))

            fig.update_layout(
                title="Economic Indicators Over Time",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error in time series plot: {e}")
            return None

    def create_surprise_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create line plot of economic surprises over time"""
        try:
            if 'value' not in df.columns or 'expected' not in df.columns:
                return go.Figure()
            
            # Calculate surprises
            df['surprise'] = df['value'] - df['expected']
            
            # Create figure
            fig = go.Figure()
            
            # Add line for each indicator
            for indicator in df['indicator'].unique():
                mask = df['indicator'] == indicator
                data = df[mask].sort_values('date')
                
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data['surprise'],
                    mode='lines+markers',
                    name=indicator,
                    line=dict(color=self.indicator_colors.get(indicator, 'gray')),
                    marker=dict(size=8),
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Surprise: %{y:.2f}<br>"
                    )
                ))
            
            fig.update_layout(
                title="Economic Surprises Over Time",
                xaxis_title="Date",
                yaxis_title="Surprise",
                hovermode='closest',
                showlegend=True
            )
            
            return fig
        except Exception as e:
            logging.error(f"Error creating surprise plot: {str(e)}")
            return go.Figure() 