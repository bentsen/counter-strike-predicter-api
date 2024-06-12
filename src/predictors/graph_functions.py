import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import logging

from sklearn.preprocessing import LabelEncoder

from config.settings import DATA_DIR
from src.predictors.evaluations import evaluate_models, evaluate_models_across_splits

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'csgo_round_snapshots.csv'))
    le = LabelEncoder()

    df['bomb_planted'] = df['bomb_planted'].astype(int)
    df = pd.get_dummies(df, columns=['map'], dtype=int)
    df["round_winner"] = le.fit_transform(df["round_winner"])

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

        # radar chart model performance comparison
        logger.info("Creating radar chart model performance comparison")
        results = evaluate_models(df)

        results.set_index('Model', inplace=True)

        labels = results.columns
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for model in results.index:
            data = results.loc[model].tolist()
            data += data[:1]
            ax.plot(angles, data, label=model, linewidth=2)
            ax.fill(angles, data, alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)

        plt.title('Model Performance Comparison')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        save_plot(fig, graph_dir, 'radar_charts', 'model_performance_comparison.png')

        # Performance Plots Across Different Split Ratios
        logger.info("Creating performance plots across different split ratios")
        results = evaluate_models_across_splits(df, [0.2, 0.3, 0.4, 0.5, 0.6])

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        models = results['Model'].unique()

        for metric in metrics:
            plt.figure(figsize=(10, 6))

            for model in models:
                model_results = results[results['Model'] == model]
                plt.plot(model_results['Test Size'], model_results[metric], marker='o', label=model)

            plt.title(f'{metric} across different split ratios')
            plt.xlabel('Test Size')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            save_plot(plt, graph_dir, 'performance_across_splits', f'{metric.lower()}_across_splits.png')

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