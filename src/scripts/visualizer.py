"""
Contains the three core analysis-and-visualization functions used in the
EDA notebook. Each function takes the merged DataFrame, performs its own
aggregation or modeling, and returns a matplotlib Figure object.

Functions
---------
plot_panic_metric          : Bar chart of Taker-trade % by sentiment regime
analyze_whales_vs_retail   : Grouped bar chart of Net PnL by trader tier × sentiment
perform_behavioral_clustering : K-Means clustering scatter (win rate vs. frequency)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Global Seaborn theme applied once at import time
sns.set_theme(style="whitegrid", palette="muted")


# Analysis I — Execution Urgency (Taker % by Sentiment)


def plot_panic_metric(df_merged: pd.DataFrame):
    """
    Plots the percentage of Taker (spread-crossing) trades within each
    Fear & Greed sentiment regime. A higher Taker % indicates more
    aggressive, panic-driven order routing.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged dataset with 'classification' and 'Crossed' columns.

    Returns
    -------
    matplotlib.figure.Figure
        The completed bar chart figure.
    """
    # Aggregate: mean of boolean 'Crossed' gives the proportion of Takers
    panic_df = df_merged.groupby("classification")["Crossed"].mean().reset_index()
    # Convert proportion (0–1) to percentage (0–100) for readability
    panic_df["Crossed"] *= 100

    # Enforce a logical left-to-right ordering from fear → greed
    sentiment_order = [
        "Extreme Fear",
        "Fear",
        "Neutral",
        "Greed",
        "Extreme Greed",
    ]

    # Build the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=panic_df,
        x="classification",
        y="Crossed",
        order=sentiment_order,
        palette="coolwarm",  # Blue (fear) → Red (greed) diverging palette
        ax=ax,
    )

    ax.set_title(
        "The Panic Metric: % of Aggressive (Taker) Trades by Market Sentiment",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Market Sentiment", fontsize=12)
    ax.set_ylabel("Taker Trade Percentage (%)", fontsize=12)

    # Annotate each bar with its exact percentage value
    for patch in ax.patches:
        # Seaborn adds non-Rectangle patches (e.g., error-bar lines); skip them
        if not isinstance(patch, Rectangle):
            continue

        height = patch.get_height()
        x_center = patch.get_x() + patch.get_width() / 2

        ax.annotate(
            f"{height:.1f}%",
            (x_center, height),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout()
    return fig


# Analysis II — Volume-Tier Segmentation (Whales vs. Retail)


def analyze_whales_vs_retail(df_merged: pd.DataFrame):
    """
    Segments accounts into three tiers based on average trade size, then
    plots total Net PnL for each tier across sentiment regimes.

    Tier definitions (percentile-based):
      - Retail:   bottom 75%
      - Mid-Size: 75th–95th percentile
      - Whales:   top 5%

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged dataset with 'Account', 'Size USD', 'classification',
        and 'Net_PnL' columns.

    Returns
    -------
    matplotlib.figure.Figure
        The completed grouped bar chart figure.
    """
    # Step 1: Compute each account's average trade size (USD)
    account_sizes = df_merged.groupby("Account")["Size USD"].mean().reset_index()

    # Step 2: Assign tiers using quantile-based binning
    # Bins represent cumulative quantile boundaries: [0, 75th, 95th, 100th]
    bins = [0, 0.75, 0.95, 1.0]
    labels = [
        "Retail (Bottom 75%)",
        "Mid-Size (Next 20%)",
        "Whales (Top 5%)",
    ]

    account_sizes["Trader_Tier"] = pd.qcut(
        account_sizes["Size USD"],
        q=bins,
        labels=labels,
    )

    # Step 3: Join tiers back onto every trade
    df_with_tiers = pd.merge(
        df_merged,
        account_sizes[["Account", "Trader_Tier"]],
        on="Account",
        how="left",
    )

    # Step 4: Aggregate total Net PnL per (sentiment × tier)
    pnl_summary = (
        df_with_tiers.groupby(["classification", "Trader_Tier"])["Net_PnL"]
        .sum()
        .reset_index()
    )

    # Enforce consistent x-axis ordering
    sentiment_order = [
        "Extreme Fear",
        "Fear",
        "Neutral",
        "Greed",
        "Extreme Greed",
    ]

    # Build the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=pnl_summary,
        x="classification",
        y="Net_PnL",
        hue="Trader_Tier",
        order=sentiment_order,
        palette="viridis",  # Perceptually uniform, colorblind-friendly
        ax=ax,
    )

    ax.set_title(
        "Who Profits? Total PnL by Trader Size across Market Sentiments",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Market Sentiment", fontsize=12)
    ax.set_ylabel("Total Net PnL (USD)", fontsize=12)
    # Zero-line highlights the break-even boundary
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.legend(title="Trader Tier")

    fig.tight_layout()
    return fig


# Analysis III — Unsupervised Behavioral Clustering (K-Means)


def perform_behavioral_clustering(df_merged: pd.DataFrame):
    """
    Discovers natural trader archetypes using K-Means clustering on three
    per-account behavioral features, then plots the result as a scatter
    of win rate vs. trade frequency.

    Clustering features (all z-score normalized):
      1. Total_Trades   — activity intensity
      2. Avg_Size_USD   — capital conviction per trade
      3. Win_Rate       — consistency / edge

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged dataset with 'Account', 'Trade ID', 'Size USD', and
        'Net_PnL' columns.

    Returns
    -------
    matplotlib.figure.Figure
        The completed scatter plot figure.
    """
    # Step 1: Build per-account profile table
    trader_profiles = df_merged.groupby("Account").agg(
        Total_Trades=("Trade ID", "count"),  # How often they trade
        Avg_Size_USD=("Size USD", "mean"),  # How large each trade is
        Winning_Trades=(
            "Net_PnL",
            lambda x: (x > 0).sum(),
        ),  # Count of profitable trades
    )

    # Derive win rate: proportion of trades with positive Net PnL
    trader_profiles["Win_Rate"] = (
        trader_profiles["Winning_Trades"] / trader_profiles["Total_Trades"]
    )

    # Step 2: Standardize features for fair distance computation
    # K-Means uses Euclidean distance, so features must be on the same scale.
    # StandardScaler applies z-score normalization: (x - mean) / std
    features = trader_profiles[["Total_Trades", "Avg_Size_USD", "Win_Rate"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Step 3: Fit K-Means with k=3 clusters
    # random_state=42 ensures reproducible cluster assignments across runs
    kmeans = KMeans(n_clusters=3, random_state=42)
    trader_profiles["Cluster"] = kmeans.fit_predict(scaled_features)

    # Step 4: Map cluster IDs to human-readable archetype labels
    # These labels were assigned after inspecting the cluster centroids and
    # their behavioral characteristics in exploratory iterations.
    cluster_mapping = {
        0: "High-Frequency Losers",
        1: "Patient Snipers",
        2: "Volume Whales",
    }

    trader_profiles["Behavioral_Profile"] = trader_profiles["Cluster"].map(
        cluster_mapping
    )

    # Build the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=trader_profiles,
        x="Total_Trades",
        y="Win_Rate",
        hue="Behavioral_Profile",
        palette="Set1",  # High-contrast qualitative palette
        alpha=0.7,  # Slight transparency to reveal density
        ax=ax,
    )

    ax.set_title(
        "Trader Clustering: Win Rate vs. Trade Frequency",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Total Trades (Log Scale)", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    # Log scale on x-axis prevents the high-frequency tail from compressing
    # the majority of data points into an unreadable cluster on the left
    ax.set_xscale("log")

    fig.tight_layout()
    return fig
