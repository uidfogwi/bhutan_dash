# -*- coding: utf-8 -*-
from dash import dcc, html
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO
from collections import Counter
from functools import lru_cache
import os

# Define file paths
file_paths = {
    '2020_part1': 'data/bhutan_2020_p1.csv',
    '2020_part2': 'data/bhutan_2020_p2.csv',
    '2020_part3': 'data/bhutan_2020_p3.csv',
    '2020_part4': 'data/bhutan_2020_p4.csv',
    '2021_part1': 'data/bhutan_2021_p1.csv',
    '2021_part2': 'data/bhutan_2021_p2.csv',
    '2022_part1': 'data/bhutan_2022_p1.csv',
    '2022_part2': 'data/bhutan_2022_p2.csv',
    '2023_part1': 'data/bhutan_2023_p1.csv',
    '2023_part2': 'data/bhutan_2023_p2.csv',
    '2024_part1': 'data/bhutan_2024_p1.csv',
    '2024_part2': 'data/bhutan_2024_p2.csv'
}


# Function to load datasets
@lru_cache(maxsize=32)  # Cache the last 32 calls
def load_data(year=None):
    try:
        if year == '2020':
            df1 = pd.read_csv(file_paths['2020_part1'])
            df2 = pd.read_csv(file_paths['2020_part2'])
            df3 = pd.read_csv(file_paths['2020_part3'])  # New part
            df4 = pd.read_csv(file_paths['2020_part4'])  # New part
            df = pd.concat([df1, df2, df3, df4], ignore_index=True)  # Concatenate all parts
        elif year == '2021':
            df1 = pd.read_csv(file_paths['2021_part1'])
            df2 = pd.read_csv(file_paths['2021_part2'])
            df = pd.concat([df1, df2], ignore_index=True)
        elif year == '2022':
            df1 = pd.read_csv(file_paths['2022_part1'])
            df2 = pd.read_csv(file_paths['2022_part2'])
            df = pd.concat([df1, df2], ignore_index=True)
        elif year == '2023':
            df1 = pd.read_csv(file_paths['2023_part1'])
            df2 = pd.read_csv(file_paths['2023_part2'])
            df = pd.concat([df1, df2], ignore_index=True)
        elif year == '2024':
            df1 = pd.read_csv(file_paths['2024_part1'])
            df2 = pd.read_csv(file_paths['2024_part2'])
            df = pd.concat([df1, df2], ignore_index=True)
        elif year and year in file_paths:
            # Load data for single-part years (if any)
            df = pd.read_csv(file_paths[year])
        else:
            # Load and concatenate data from all years
            dataframes = []
            for year_key in file_paths:
                df_part = pd.read_csv(file_paths[year_key])
                dataframes.append(df_part)
            df = pd.concat(dataframes, ignore_index=True)
        
        # Rename columns to ensure consistency in the DataFrame
        df = df.rename(columns={'V2Tone': 'Tone', 'FinalThemes': 'Theme'})
        
        # Calculate the count of each theme
        df['Count'] = df.groupby('Theme')['Theme'].transform('count')
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


# Function to truncate labels
def truncate_labels(label):
    return (label[:30] + '...') if len(label) > 30 else label

# Function to extract and count keywords from the Themes column
def extract_keywords(df):
    all_keywords = []
    for theme_list in df['Theme'].dropna().str.split(';'):
        all_keywords.extend(theme_list)
    return pd.DataFrame(Counter(all_keywords).items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

# Function to generate word cloud
def generate_wordcloud(text):
    additional_stopwords = set(['http', 'https', 'html', 'com', 'org', 'net', 'news', 'article', 'local'])
    stopwords = STOPWORDS.union(additional_stopwords)
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
    img = BytesIO()
    wc.to_image().save(img, format='PNG')
    return base64.b64encode(img.getvalue()).decode()

# Initialize the app with Bootstrap CSS
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Navbar component
navbar = dbc.NavbarSimple(
    brand="Bhutan News Data Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Footer component
footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                "© 2024 Bhutan News Dashboard | Developed by Paul Sheng",
                className="text-center",
                style={"padding": "20px"}
            )
        )
    ),
    className="mt-5"
)

# Layout for the dashboard
app.layout = html.Div([
    navbar,
    dbc.Container([
        html.H1("Bhutan News Data Analysis", className="text-center mb-4"),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='year-selector',
                    options=[{'label': str(year), 'value': str(year)} for year in file_paths.keys()] + [{'label': 'All Years', 'value': 'all'}],
                    value='all',
                    multi=False,
                    placeholder="Select Year"
                ),
                dcc.Tabs([
                    dcc.Tab(label='Bar Chart', children=[dcc.Graph(id='bar-chart')]),
                    dcc.Tab(label='Treemap', children=[dcc.Graph(id='treemap')]),
                    dcc.Tab(label='Word Cloud', children=[html.Img(id='wordcloud')]),
                    dcc.Tab(label='Keyword Count', children=[dcc.Graph(id='keyword-count')]),
                    dcc.Tab(label='Heatmap', children=[dcc.Graph(id='heatmap')]),
                    dcc.Tab(label='Pie Chart', children=[dcc.Graph(id='pie-chart')]),
                    dcc.Tab(label='Box Plot', children=[dcc.Graph(id='box-plot')]),
                ])
            ])
        ])
    ], fluid=True),
    footer
])

# Callbacks

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('year-selector', 'value')]
)
def update_bar_chart(selected_year):
    df = load_data(selected_year if selected_year != 'all' else None)
    if df.empty or 'Theme' not in df.columns:
        return go.Figure()
    
    top_themes = df['Theme'].value_counts().nlargest(10).reset_index()
    top_themes.columns = ['Theme', 'Count']
    top_themes['Theme'] = top_themes['Theme'].apply(truncate_labels)
    fig = px.bar(top_themes, x='Theme', y='Count', title='Top 10 Themes')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

@app.callback(
    Output('treemap', 'figure'),
    [Input('year-selector', 'value')]
)
def update_treemap(selected_year):
    df = load_data(selected_year if selected_year != 'all' else None)
    if df.empty:
        return go.Figure()
    
    fig = px.treemap(df, path=['Theme'], values='Count', title='Treemap of Themes')
    return fig

@app.callback(
    Output('wordcloud', 'src'),
    [Input('year-selector', 'value')]
)
def update_wordcloud(selected_year):
    df = load_data(selected_year if selected_year != 'all' else None)
    if df.empty:
        return ""
    
    text = ' '.join(df['DocumentIdentifier'].dropna())
    return f"data:image/png;base64,{generate_wordcloud(text)}"

@app.callback(
    Output('keyword-count', 'figure'),
    [Input('year-selector', 'value')]
)
def update_keyword_count(selected_year):
    df = load_data(selected_year if selected_year != 'all' else None)
    if df.empty:
        return go.Figure()

    keyword_df = extract_keywords(df).head(10)
    
    if keyword_df.empty:
        return go.Figure()

    fig = px.bar(keyword_df, x='Keyword', y='Count', title=f'Top 10 Keywords for {selected_year}')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

@app.callback(
    Output('heatmap', 'figure'),
    [Input('year-selector', 'value')]
)
def update_heatmap(selected_year):
    df = load_data(selected_year if selected_year != 'all' else None)
    
    if df.empty or 'Tone' not in df.columns or 'Theme' not in df.columns:
        return go.Figure()
    
    # Extract keywords from the Theme column
    df_exploded = df.assign(Keyword=df['Theme'].str.split(';')).explode('Keyword')
    df_exploded['Tone'] = pd.to_numeric(df_exploded['Tone'], errors='coerce')
    df_exploded = df_exploded.dropna(subset=['Tone'])
    
    # Coarser bins for Tone ranges
    df_exploded['Tone Range'] = pd.cut(df_exploded['Tone'], bins=[-5, -2, 0, 2, 5], labels=['<-2', '-2 to 0', '0 to 2', '>2'])
    
    # Limit to top 10 most frequent keywords for clarity
    top_keywords = df_exploded['Keyword'].value_counts().nlargest(10).index
    df_exploded = df_exploded[df_exploded['Keyword'].isin(top_keywords)]
    
    # Shorten or abbreviate keywords for better visualization
    df_exploded['Keyword'] = df_exploded['Keyword'].apply(lambda x: (x[:20] + '...') if len(x) > 20 else x)
    
    heatmap_data = df_exploded.pivot_table(index='Tone Range', columns='Keyword', aggfunc='size', fill_value=0)
    
    if heatmap_data.empty:
        return go.Figure()
    
    # Increase figure size and font size for better readability
    fig = px.imshow(heatmap_data, 
                    color_continuous_scale='Viridis',
                    aspect='auto', 
                    title=f'Heatmap of V2Tone by Keyword for {selected_year}')
    
    # Adjust layout for larger visualization and improved readability
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=12)),  # Rotate and increase font size
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=150, r=150, t=100, b=150),  # Increase margins
        width=1200,  # Increase width further
        height=800,  # Keep height
        coloraxis_colorbar=dict(title='Count', tickfont=dict(size=12))  # Customize colorbar
    )
    
    return fig




# Function to abbreviate keywords to the first three words
def abbreviate_keyword(keyword):
    words = keyword.split('_')
    if len(words) > 3:
        return '_'.join(words[:3]) + '...'
    return keyword

# Updated Pie Chart Callback
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('year-selector', 'value')]
)
def update_pie_chart(selected_year):
    df = load_data(selected_year if selected_year != 'all' else None)
    
    if df.empty or 'Theme' not in df.columns:
        print(f"Pie Chart: DataFrame is empty or missing Theme column for year {selected_year}.")
        return go.Figure()

    keyword_df = extract_keywords(df).head(10)
    
    if keyword_df.empty:
        print(f"Pie Chart: No keywords found for year {selected_year}.")
        return go.Figure()

    # Abbreviate the keywords for the pie chart
    keyword_df['Abbreviated'] = keyword_df['Keyword'].apply(abbreviate_keyword)

    # Create the pie chart
    fig = px.pie(
        keyword_df, 
        names='Abbreviated',  # Use abbreviated keywords
        values='Count', 
        title=f'Keyword Distribution for {selected_year}',
        hole=0.3  # Optional: Makes a donut chart for better label visibility
    )
    
    # Adjust layout to avoid overlap and improve readability
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",  # Change to vertical orientation
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1  # Move the legend to the right side of the chart
        ),
        margin=dict(t=40, b=40, l=40, r=120),  # Increase right margin to accommodate legend
        height=400,  # Increase the height of the pie chart for better visibility
    )

    # Ensure labels do not overlap and are fully visible
    fig.update_traces(
        textposition='inside',  # Place labels inside the slices
        textinfo='percent+label',  # Show percentage and label inside the slice
        insidetextorientation='radial',  # Ensure text is oriented properly within the slice
        textfont_size=12  # Slightly reduce text size to fit better
    )
    
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    [Input('year-selector', 'value')]
)
def update_box_plot(selected_year):
    df = load_data(selected_year if selected_year != 'all' else None)
    
    if df.empty or 'Tone' not in df.columns or 'Theme' not in df.columns:
        return go.Figure()
    
    # Extract keywords from the Theme column
    keyword_df = extract_keywords(df)
    
    if keyword_df.empty:
        return go.Figure()
    
    # Filter dataframe to include only rows with top 10 keywords for better clarity
    top_keywords = keyword_df['Keyword'].head(10).values
    df_filtered = df[df['Theme'].apply(lambda x: any(k in x for k in top_keywords))]

    # Explode the themes column to handle multiple keywords in a single row
    df_filtered = df_filtered.assign(Keyword=df_filtered['Theme'].str.split(';')).explode('Keyword')

    # Filter only the top keywords
    df_filtered = df_filtered[df_filtered['Keyword'].isin(top_keywords)]
    
    # Abbreviate long keywords
    df_filtered['Keyword'] = df_filtered['Keyword'].apply(lambda x: (x[:20] + '...') if len(x) > 20 else x)
    
    # Sort keywords by median Tone value
    keyword_order = df_filtered.groupby('Keyword')['Tone'].median().sort_values().index
    
    # Create the box plot
    fig = px.box(df_filtered, 
                 x='Keyword', 
                 y='Tone', 
                 title=f'Tone Distribution by Keywords for {selected_year}',
                 category_orders={'Keyword': keyword_order})
    
    # Customize the layout
    fig.update_layout(
        xaxis_title="Keyword",
        yaxis_title="Tone",
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=40, b=150),  # Increase bottom margin for more space
        width=1200,  # Increase width for better readability
        height=600  # Adjust height
    )
    
    # Option to hide outliers
    fig.update_traces(boxpoints='outliers')
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
























                                                












