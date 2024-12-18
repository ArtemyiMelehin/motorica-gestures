# Работа с табличными данными
import pandas as pd


# --------------------------------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ ДАННЫХ

import plotly.express as px
import plotly.io as pio


def fig_montage(
        fig_data: pd.DataFrame,
        title: str = '', 
        width: int = 1000, 
        height: int = 600,
        mult_labels: int = 1_000_000,
        **extra_labels
        ):

    fig_data = pd.DataFrame(fig_data)

    for extra_label in extra_labels:
        fig_data[extra_label] = extra_labels[extra_label] * mult_labels
        
    fig = px.line(fig_data, width=width, height=height, title=title)

    line_width = 1 if pio.templates.default == 'plotly_dark' else 2
    fig.update_traces(line=dict(width=line_width))
    return fig