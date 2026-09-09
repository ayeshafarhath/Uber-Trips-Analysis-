import plotly.express as px
import pandas as pd

def plot_user_pins(pins, labels, centroids, title="Frequent Pins"):
    df = pd.DataFrame(pins, columns=['lat','lon'])
    df['cluster'] = labels.astype(str)
    fig = px.scatter_mapbox(df, lat='lat', lon='lon', color='cluster',
                            zoom=12, height=600, title=title)
    fig.update_layout(mapbox_style="carto-positron")
    # Add centroids
    if len(centroids)>0:
        c_df = pd.DataFrame(centroids, columns=['lat','lon'])
        fig.add_scattermapbox(lat=c_df['lat'], lon=c_df['lon'], 
                              mode='markers', marker=dict(size=14, color='black', symbol='star'),
                              name='centroid')
    return fig
