import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

from config.settings import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'csgo_round_snapshots.csv'))

    # Ensure correct data types
    df['time_left'] = pd.to_numeric(df['time_left'], errors='coerce')
    df['ct_score'] = pd.to_numeric(df['ct_score'], errors='coerce')
    df['t_score'] = pd.to_numeric(df['t_score'], errors='coerce')
    df['ct_health'] = pd.to_numeric(df['ct_health'], errors='coerce')
    df['t_health'] = pd.to_numeric(df['t_health'], errors='coerce')
    df['ct_money'] = pd.to_numeric(df['ct_money'], errors='coerce')
    df['t_money'] = pd.to_numeric(df['t_money'], errors='coerce')
    df['ct_armor'] = pd.to_numeric(df['ct_armor'], errors='coerce')
    df['t_armor'] = pd.to_numeric(df['t_armor'], errors='coerce')
    df['bomb_planted'] = df['bomb_planted'].astype(int)

    # Convert round_winner to numeric
    df['round_winner'] = df['round_winner'].map({'CT': 1, 'T': 0})

    # Grenades columns
    grenade_columns = [col for col in df.columns if 'grenade' in col]
    for col in grenade_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def save_plot(fig, graph_dir, chart_type, filename):
    path = os.path.join(graph_dir, chart_type)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, filename))


def generate_graphs():
    try:
        df = load_data()
        graph_dir = os.path.join('src', 'graphs', 'round_predictor')

        # Aggregate data by 10-second intervals
        df_agg = df.groupby(df['time_left'] // 10 * 10).mean(numeric_only=True)

        # Score Progression Over Time
        logger.info("Creating score progression over time graph")
        fig, ax = plt.subplots()
        ax.plot(df_agg.index, df_agg['ct_score'], label='CT Score')
        ax.plot(df_agg.index, df_agg['t_score'], label='T Score')
        ax.set_xlabel('Time Left')
        ax.set_ylabel('Score')
        ax.set_title('Score Progression Over Time')
        ax.legend()
        save_plot(fig, graph_dir, 'line_charts', 'score_progression_over_time.png')
        plt.close(fig)

        # Health vs. Time
        logger.info("Creating health vs. time graph")
        fig, ax = plt.subplots()
        ax.plot(df_agg.index, df_agg['ct_health'], label='CT Health')
        ax.plot(df_agg.index, df_agg['t_health'], label='T Health')
        ax.set_xlabel('Time Left')
        ax.set_ylabel('Health')
        ax.set_title('Health vs. Time')
        ax.legend()
        save_plot(fig, graph_dir, 'line_charts', 'health_vs_time.png')
        plt.close(fig)

        # Money vs. Time
        logger.info("Creating money vs. time graph")
        fig, ax = plt.subplots()
        ax.plot(df_agg.index, df_agg['ct_money'], label='CT Money')
        ax.plot(df_agg.index, df_agg['t_money'], label='T Money')
        ax.set_xlabel('Time Left')
        ax.set_ylabel('Money')
        ax.set_title('Money vs. Time')
        ax.legend()
        save_plot(fig, graph_dir, 'line_charts', 'money_vs_time.png')
        plt.close(fig)

        # Grenade Usage Over Time
        logger.info("Creating grenade usage over time graph")
        fig, ax = plt.subplots()
        ax.plot(df_agg.index, df_agg['ct_grenade_flashbang'], label='CT Flashbang')
        ax.plot(df_agg.index, df_agg['t_grenade_smokegrenade'], label='T Smoke Grenade')
        ax.set_xlabel('Time Left')
        ax.set_ylabel('Number of Grenades')
        ax.set_title('Grenade Usage Over Time')
        ax.legend()
        save_plot(fig, graph_dir, 'line_charts', 'grenade_usage_over_time.png')
        plt.close(fig)

        # Armor Levels Over Time
        logger.info("Creating armor levels over time graph")
        fig, ax = plt.subplots()
        ax.plot(df_agg.index, df_agg['ct_armor'], label='CT Armor')
        ax.plot(df_agg.index, df_agg['t_armor'], label='T Armor')
        ax.set_xlabel('Time Left')
        ax.set_ylabel('Armor')
        ax.set_title('Armor Levels Over Time')
        ax.legend()
        save_plot(fig, graph_dir, 'line_charts', 'armor_levels_over_time.png')
        plt.close(fig)

        # Bomb Planting Events
        logger.info("Creating bomb planting events graph")
        fig, ax = plt.subplots()
        ax.scatter(df['time_left'], df['bomb_planted'])
        ax.set_xlabel('Time Left')
        ax.set_ylabel('Bomb Planted')
        ax.set_title('Bomb Planting Events')
        save_plot(fig, graph_dir, 'scatter_plots', 'bomb_planting_events.png')
        plt.close(fig)

        # Round Winners by Map
        logger.info("Creating round winners by map graph")
        fig, ax = plt.subplots()
        map_wins = df.groupby('map')['round_winner'].sum()
        map_wins.plot(kind='bar', ax=ax)
        ax.set_xlabel('Map')
        ax.set_ylabel('Number of Rounds Won by CT')
        ax.set_title('Round Winners by Map')
        save_plot(fig, graph_dir, 'bar_charts', 'round_winners_by_map.png')
        plt.close(fig)

        # Equipment Value Over Time with Aggregation
        logger.info("Creating equipment value over time graph with aggregation")
        fig, ax = plt.subplots()
        ax.plot(df_agg.index, df_agg['ct_money'], label='CT Equipment Value')
        ax.plot(df_agg.index, df_agg['t_money'], label='T Equipment Value')
        ax.set_xlabel('Time Left')
        ax.set_ylabel('Equipment Value')
        ax.set_title('Equipment Value Over Time')
        ax.legend()
        save_plot(fig, graph_dir, 'line_charts', 'equipment_value_over_time.png')
        plt.close(fig)

        # Round Outcome by Health and Armor
        logger.info("Creating round outcome by health and armor graph")
        fig, ax = plt.subplots()
        sns.boxplot(x='round_winner', y='ct_health', data=df, ax=ax)
        sns.boxplot(x='round_winner', y='t_health', data=df, ax=ax)
        sns.boxplot(x='round_winner', y='ct_armor', data=df, ax=ax)
        sns.boxplot(x='round_winner', y='t_armor', data=df, ax=ax)
        ax.set_xlabel('Round Winner')
        ax.set_ylabel('Health and Armor')
        ax.set_title('Round Outcome by Health and Armor')
        save_plot(fig, graph_dir, 'box_plots', 'round_outcome_by_health_and_armor.png')
        plt.close(fig)

        # Winning Probability by Round Time
        logger.info("Creating winning probability by round time graph")
        fig, ax = plt.subplots()
        heatmap_data = pd.pivot_table(df, values='round_winner', index='time_left', columns='ct_score', aggfunc='mean')
        sns.heatmap(heatmap_data, ax=ax, cmap="YlGnBu")
        ax.set_xlabel('CT Score')
        ax.set_ylabel('Time Left')
        ax.set_title('Winning Probability by Round Time')
        save_plot(fig, graph_dir, 'heatmaps', 'winning_probability_by_round_time.png')
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating graphs: {e}")
        raise