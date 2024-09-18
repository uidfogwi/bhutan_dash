# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO
from collections import Counter
from functools import lru_cache

# Define file paths
file_paths = {
    '2020_part1': r'data/bhutan_2020_p1.csv',
    '2020_part2': r'data/bhutan_2020_p2.csv',
    '2020_part3': r'data/bhutan_2020_p3.csv',
    '2020_part4': r'data/bhutan_2020_p4.csv',
    '2021_part1': r'data/bhutan_2021_p1.csv',
    '2021_part2': r'data/bhutan_2021_p2.csv',
    '2022_part1': r'data/bhutan_2022_p1.csv',
    '2022_part2': r'data/bhutan_2022_p2.csv',
    '2023_part1': r'data/bhutan_2023_p1.csv',
    '2023_part2': r'data/bhutan_2023_p2.csv',
    '2024_part1': r'data/bhutan_2024_p1.csv',
    '2024_part2': r'data/bhutan_2024_p2.csv'
}

# Function to load datasets
@lru_cache(maxsize=32)
def load_data(year=None):
    try:
        if year == '2020':
            df1 = pd.read_csv(file_paths['2020_part1'])
            df2 = pd.read_csv(file_paths['2020_part2'])
            df3 = pd.read_csv(file_paths['2020_part3'])
            df4 = pd.read_csv(file_paths['2020_part4'])
            df = pd.concat([df1, df2, df3, df4], ignore_index=True)
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
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to generate word cloud
def generate_wordcloud(text):
    additional_stopwords = set(['http', 'https', 'html', 'com', 'org', 'net', 'news', 'article', 'local'])
    stopwords = STOPWORDS.union(additional_stopwords)
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
    img = BytesIO()
    wc.to_image().save(img, format='PNG')
    return base64.b64encode(img.getvalue()).decode()

# Streamlit App Layout
st.title("Bhutan News Data Dashboard")

# Sidebar for year selection
selected_year = st.sidebar.selectbox("Select Year", options=list(file_paths.keys()) + ['all'], index=len(file_paths))

# Load data
df = load_data(selected_year if selected_year != 'all' else None)

# Ensure that data is not empty
if df.empty:
    st.warning("No data available for the selected year.")
else:
    # Top 10 Themes Bar Chart
    st.subheader("Top 10 Themes")
    top_themes = df['Theme'].value_counts().nlargest(10).reset_index()
    top_themes.columns = ['Theme', 'Count']
    fig_bar = px.bar(top_themes, x='Theme', y='Count', title='Top 10 Themes')
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar)

    # Treemap
    st.subheader("Treemap of Themes")
    fig_treemap = px.treemap(df, path=['Theme'], values='Count', title='Treemap of Themes')
    st.plotly_chart(fig_treemap)

    # Word Cloud
    st.subheader("Word Cloud")
    if 'DocumentIdentifier' in df.columns:
        text = ' '.join(df['DocumentIdentifier'].dropna())
        st.image(f"data:image/png;base64,{generate_wordcloud(text)}", use_column_width=True)

    # Top 10 Keywords Bar Chart
    st.subheader("Top 10 Keywords")
    def extract_keywords(df):
        all_keywords = []
        for theme_list in df['Theme'].dropna().str.split(';'):
            all_keywords.extend(theme_list)
        return pd.DataFrame(Counter(all_keywords).items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

    keyword_df = extract_keywords(df).head(10)
    fig_keywords = px.bar(keyword_df, x='Keyword', y='Count', title=f'Top 10 Keywords for {selected_year}')
    fig_keywords.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_keywords)

    # Heatmap
    st.subheader("Heatmap of V2Tone by Keyword")
    if 'Tone' in df.columns and 'Theme' in df.columns:
        df_exploded = df.assign(Keyword=df['Theme'].str.split(';')).explode('Keyword')
        df_exploded['Tone'] = pd.to_numeric(df_exploded['Tone'], errors='coerce')
        df_exploded = df_exploded.dropna(subset=['Tone'])

        # Coarser bins for Tone ranges
        df_exploded['Tone Range'] = pd.cut(df_exploded['Tone'], bins=[-5, -2, 0, 2, 5], labels=['<-2', '-2 to 0', '0 to 2', '>2'])

        # Limit to top 10 most frequent keywords for clarity
        top_keywords = df_exploded['Keyword'].value_counts().nlargest(10).index
        df_exploded = df_exploded[df_exploded['Keyword'].isin(top_keywords)]

        heatmap_data = df_exploded.pivot_table(index='Tone Range', columns='Keyword', aggfunc='size', fill_value=0)

        fig_heatmap = px.imshow(heatmap_data, color_continuous_scale='Viridis', title=f'Heatmap for {selected_year}')
        st.plotly_chart(fig_heatmap)

    # Pie Chart
    st.subheader("Keyword Distribution")
    fig_pie = px.pie(keyword_df, names='Keyword', values='Count', title=f'Keyword Distribution for {selected_year}')
    st.plotly_chart(fig_pie)

    # Box Plot
    st.subheader("Tone Distribution by Keywords")
    if 'Tone' in df.columns and 'Theme' in df.columns:
        df_filtered = df[df['Theme'].apply(lambda x: any(k in x for k in keyword_df['Keyword'].values))]
        df_filtered = df_filtered.assign(Keyword=df_filtered['Theme'].str.split(';')).explode('Keyword')
        fig_box = px.box(df_filtered, x='Keyword', y='Tone', title=f'Tone Distribution by Keywords for {selected_year}')
        st.plotly_chart(fig_box)


















                                                












