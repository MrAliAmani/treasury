import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class FinancialVisualizer:
    """Class to create interactive financial visualizations using Plotly"""
    
    def __init__(self):
        self.indicator_colors = {
            "Inflation": "red",
            "Growth": "blue", 
            "Unemployment": "green"
        }

    def create_surprise_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Create scatter plot with regression line"""
        if df is None or df.empty or 'surprise' not in df.columns or 'yield_change' not in df.columns:
            return go.Figure()  # Return empty figure
        
        # Ensure we have valid data for the fit
        mask = df['surprise'].notna() & df['yield_change'].notna()
        valid_data = df[mask]
        
        if len(valid_data) < 2:  # Need at least 2 points for regression
            return go.Figure()
        
        try:
            # Calculate regression
            z = np.polyfit(valid_data['surprise'], valid_data['yield_change'], 1)
            p = np.poly1d(z)
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=valid_data['surprise'],
                y=valid_data['yield_change'],
                mode='markers',
                name='Data Points'
            ))
            
            # Add regression line
            x_range = np.linspace(valid_data['surprise'].min(), valid_data['surprise'].max(), 100)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name='Regression Line'
            ))
            
            fig.update_layout(
                title='Economic Surprises vs Yield Changes',
                xaxis_title='Surprise',
                yaxis_title='Yield Change (%)'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            return go.Figure()  # Return empty figure on error

    def create_yield_curve_animation(self, 
                                   maturities: List[int],
                                   pre_curves: List[List[float]], 
                                   post_curves: List[List[float]]) -> go.Figure:
        """
        Create animated yield curve comparison
        
        Args:
            maturities: List of maturities in years
            pre_curves: List of pre-event yield curves
            post_curves: List of post-event yield curves
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add initial curves
        fig.add_trace(go.Scatter(
            x=maturities,
            y=pre_curves[0],
            mode='lines+markers',
            name='Pre-Event',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=maturities,
            y=post_curves[0],
            mode='lines+markers', 
            name='Post-Event',
            line=dict(color='purple')
        ))

        # Create frames for animation
        frames = []
        for i in range(len(pre_curves)):
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(x=maturities, y=pre_curves[i]),
                        go.Scatter(x=maturities, y=post_curves[i])
                    ],
                    name=str(i)
                )
            )
            
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title="Dynamic Yield Curve Comparison",
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield (%)",
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 800}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0}}]),
                ]
            )]
        )
        
        return fig

    def create_impact_heatmap(self, impact_matrix: np.ndarray,
                            indicators: List[str],
                            maturities: List[str]) -> go.Figure:
        """
        Create heatmap of surprise impacts
        
        Args:
            impact_matrix: 2D numpy array of impact values
            indicators: List of indicator names
            maturities: List of maturity labels
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=impact_matrix,
            x=maturities,
            y=indicators,
            colorscale='RdBu',
            reversescale=True,
            colorbar=dict(title="Impact")
        ))

        fig.update_layout(
            title="Heatmap of Surprise Impacts",
            xaxis_title="Yield Maturities",
            yaxis_title="Indicators"
        )
        
        return fig

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