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

    def create_intraday_surprise_analysis(self, df: pd.DataFrame, intraday_df: pd.DataFrame = None) -> go.Figure:
        """Create detailed analysis of market movement around economic surprises"""
        try:
            if df is None or df.empty or 'value' not in df.columns:
                return None

            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            for indicator in df['indicator'].unique():
                data = df[df['indicator'] == indicator].copy()
                
                # Calculate surprise
                if 'expected' in data.columns:
                    data['surprise'] = data['value'] - data['expected']
                else:
                    data['surprise'] = data['value'].diff()
                
                # Find significant surprises (>1 std dev)
                std_surprise = data['surprise'].std()
                significant_dates = data[abs(data['surprise']) > std_surprise]['date']
                
                # Plot surprise points
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=data['surprise'],
                        mode='markers',
                        name=f"{indicator} Surprises",
                        marker=dict(
                            size=10,
                            color=self.indicator_colors.get(indicator, 'gray'),
                            symbol='diamond'
                        ),
                        hovertemplate=(
                            "<b>%{x}</b><br>" +
                            f"{indicator}<br>" +
                            "Surprise: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    )
                )
                
                # Add intraday price movement if available
                if intraday_df is not None:
                    for surprise_date in significant_dates:
                        # Get intraday data around the surprise
                        date_str = surprise_date.strftime('%Y-%m-%d')
                        intraday_mask = (intraday_df['date'].dt.strftime('%Y-%m-%d') == date_str)
                        intraday_data = intraday_df[intraday_mask]
                        
                        if not intraday_data.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=intraday_data['date'],
                                    y=intraday_data['price'],
                                    mode='lines',
                                    name=f"Intraday {date_str}",
                                    line=dict(dash='dot'),
                                    secondary_y=True
                                )
                            )

            # Update layout
            fig.update_layout(
                title="Economic Surprises with Intraday Market Movement",
                xaxis_title="Date",
                yaxis_title="Surprise Value",
                yaxis2_title="Market Price",
                height=600,
                showlegend=True,
                # Add vertical lines for significant surprises
                shapes=[
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=date,
                        x1=date,
                        y0=0,
                        y1=1,
                        line=dict(color="gray", width=1, dash="dash")
                    )
                    for date in significant_dates
                ]
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error in intraday analysis: {e}")
            return None

    def create_surprise_detail_view(self, df: pd.DataFrame, date: str, window_hours: int = 2) -> go.Figure:
        """Create detailed view of market movement around a specific economic surprise"""
        try:
            if df is None or df.empty:
                return None
            
            # Convert date string to datetime if needed
            surprise_date = pd.to_datetime(date)
            
            # Get data for the specific date
            day_data = df[df['date'].dt.date == surprise_date.date()].copy()
            
            if day_data.empty:
                return None
            
            # Create figure
            fig = go.Figure()
            
            for indicator in day_data['indicator'].unique():
                indicator_data = day_data[day_data['indicator'] == indicator]
                
                # Plot actual value
                fig.add_trace(go.Scatter(
                    x=[indicator_data['date'].iloc[0]],
                    y=[indicator_data['value'].iloc[0]],
                    mode='markers',
                    name=f"{indicator} Actual",
                    marker=dict(
                        size=12,
                        symbol='star',
                        color=self.indicator_colors.get(indicator, 'gray')
                    )
                ))
                
                # Plot expected value if available
                if 'expected' in indicator_data.columns:
                    fig.add_trace(go.Scatter(
                        x=[indicator_data['date'].iloc[0]],
                        y=[indicator_data['expected'].iloc[0]],
                        mode='markers',
                        name=f"{indicator} Expected",
                        marker=dict(
                            size=12,
                            symbol='circle',
                            color=self.indicator_colors.get(indicator, 'gray')
                        )
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"Economic Release Detail - {surprise_date.strftime('%Y-%m-%d')}",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400,
                showlegend=True,
                # Add release time marker
                shapes=[dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=surprise_date,
                    x1=surprise_date,
                    y0=0,
                    y1=1,
                    line=dict(color="red", width=2)
                )]
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error in detail view: {e}")
            return None 