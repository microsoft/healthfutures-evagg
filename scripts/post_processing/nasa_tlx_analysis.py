""" This script calculates and analyzes NASA task load index (TLX) scores for the Evidence Aggregator user study.
It compares the task workload of users with and without Copilot across 6 categories*.
Specifically, this script calculates
A) the NASA TLX weighted scores per user and per category,
B) the NASA TLX raw scores per user and per category,
C) and the NASA TLX coefficients per category.

* Categories: Mental Demand, Physical Demand, Time Pressure, Personal Performance Success, Effort Level, Frustration
  Level

The script performs the following 5 tasks:
1. Calculates and plots NASA TLX weighted and raw scores for each user, with and without Copilot.
2. Analyzes the distribution of percent changes in scores to assess normality and suitability for paired t-tests.
3. Conducts statistical tests to identify significant differences between users with decreased
   v.s. increased task load.
4. Calculates and plots NASA TLX raw and weighted scores for 6 categories to identify possible significant changes
   due to Copilot use.
5. Creates scatter plots to compare NASA TLX raw and weighted scores and coefficients across categories, identifying
   potential relationships.
"""

# Libraries
import argparse
import os
import warnings
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_rel, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.formula.api import mixedlm

# Suppressing ConvergenceWarning from statsmodels
warnings.filterwarnings("ignore", category=UserWarning)


def convert_text_to_score(text):
    """Convert a Likert scale text to a raw value."""
    score_map = {"Very Low": 5, "Low": 25, "Moderate": 50, "High": 75, "Very High": 100}
    if text not in score_map:
        raise ValueError(f"Error: Invalid text score '{text}'")
    return score_map[text]


def count_instances(df, category, columns):
    """Count the number of instances of a specific category in the columns of interest."""
    return df[columns].apply(lambda row: (row == category).sum(), axis=1)


def calculate_overall_nasa_tlx_weighted_score(df):
    """Calculate the overall NASA TLX weighted score for each user."""
    categories = [
        "Mental Demand",
        "Physical Demand",
        "Time Pressure",
        "Personal Performance Success",
        "Effort Level",
        "Frustration Level",
    ]
    return sum(df[f"{cat} raw"] * df[f"{cat} count"] for cat in categories) / 15


def create_bar_plot(users, without_scores, with_scores, ylabel, title, filename, bar_width, output_dir):
    """Create and save a bar plot comparing NASA TLX weighted and raw scores without and with Copilot for each user."""
    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        users - bar_width / 2, without_scores, bar_width, label="without"
    )  # Note setting bar1 to the return value of ax.bar for without Copilot
    ax.bar(
        users + bar_width / 2, with_scores, bar_width, label="with Copilot"
    )  # Note setting bar2 to the return value of ax.bar for with Copilot

    for i in range(len(users)):
        percent_change = ((with_scores[i] - without_scores[i]) / without_scores[i]) * 100
        ax.text(
            users[i],
            max(without_scores[i], with_scores[i]) + 0.75,
            f"{percent_change:.2f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            color="black",
        )

    ax.set_xlabel("User")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(users)
    ax.set_xticklabels([f"User {i}" for i in users])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_distribution(percent_changes, title, filename, color, output_dir):
    """Plot the distribution of percent changes in NASA TLX scores."""
    plt.figure(figsize=(12, 6))
    sns.histplot(percent_changes, kde=True, bins=8, color=color)
    plt.title(title)
    plt.xlabel("Percent Change")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def perform_stat_tests(with_scores, without_scores, user_indices):
    """Perform t-test and Wilcoxon signed-rank test on the selected users."""
    with_selected = [with_scores[i] for i in user_indices]
    without_selected = [without_scores[i] for i in user_indices]
    _, p_value_t = ttest_rel(with_selected, without_selected)
    _, p_value_w = wilcoxon(with_selected, without_selected)
    return p_value_t, p_value_w


def plot_average_scores(df1, df2, columns, title, ylabel, filename, output_dir):
    """Plot average weighted scores, raw scores, or coefficients for each category, without and with Copilot."""
    avg_scores_without_copilot = df1[columns].mean()
    avg_scores_with_copilot = df2[columns].mean()
    std_dev_without_copilot = df1[columns].std()
    std_dev_with_copilot = df2[columns].std()
    avg_scores = pd.DataFrame(
        {
            "Feature": columns,
            "Without_Copilot": avg_scores_without_copilot,
            "With_Copilot": avg_scores_with_copilot,
            "Std_Dev_Without_Copilot": std_dev_without_copilot,
            "Std_Dev_With_Copilot": std_dev_with_copilot,
        }
    )
    avg_scores_melted = avg_scores.melt(
        id_vars="Feature",
        value_vars=["Without_Copilot", "With_Copilot"],
        var_name="Condition",
        value_name="Average_Score",
    )

    plt.figure(figsize=(15, 10))
    sns.barplot(x="Feature", y="Average_Score", hue="Condition", data=avg_scores_melted, palette="husl", errorbar=None)

    for i, feature in enumerate(columns):
        plt.errorbar(
            i - 0.2,
            avg_scores.loc[avg_scores["Feature"] == feature, "Without_Copilot"].values[0],
            yerr=avg_scores.loc[avg_scores["Feature"] == feature, "Std_Dev_Without_Copilot"].values[0],
            fmt="none",
            c="black",
            capsize=5,
        )
        plt.errorbar(
            i + 0.2,
            avg_scores.loc[avg_scores["Feature"] == feature, "With_Copilot"].values[0],
            yerr=avg_scores.loc[avg_scores["Feature"] == feature, "Std_Dev_With_Copilot"].values[0],
            fmt="none",
            c="black",
            capsize=5,
        )

    y_max = avg_scores_melted["Average_Score"].max()
    y_offset = y_max * 0.05

    for i, column in enumerate(columns):
        _, p_value = mannwhitneyu(df1[column], df2[column])
        x = i
        y = y_max + y_offset
        plt.text(x, y, f"MWU p={p_value:.3f}", ha="center", va="bottom", color="black")

        column_formula = column.replace(" ", "_")
        df_combined = pd.concat([df1, df2]).rename(columns={column: column_formula})
        df_combined["Condition"] = ["Without_Copilot"] * len(df1) + ["With_Copilot"] * len(df2)
        model = mixedlm(f"{column_formula} ~ Condition", df_combined, groups=df_combined["User"])
        result = model.fit()
        condition_key = [key for key in result.pvalues.keys() if "Condition" in key][0]
        p_value = result.pvalues[condition_key]
        y = y_max + 2 * y_offset
        plt.text(x, y, f"MLM p={p_value:.3f}", ha="center", va="bottom", color="black")

    new_labels = [
        label.replace(" Success", "").replace(" weighted_score", "").replace("coef", "").replace(" raw", "")
        for label in columns
    ]
    plt.xticks(ticks=range(len(columns)), labels=new_labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Feature")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def create_combined_scatter_plot(df1, df2, columns, title, filename, output_dir):
    """Create scatter plots to compare the various NASA TLX scores
    (raw, weighted, and coefficients) for each pair of categories."""
    combinations_list = list(combinations(columns, 2))
    plt.figure(figsize=(25, 15))

    for i, (cat1, cat2) in enumerate(combinations_list):
        x1, y1 = df1[cat1], df1[cat2]
        x2, y2 = df2[cat1], df2[cat2]

        model1 = LinearRegression().fit(x1.values.reshape(-1, 1), y1)
        model2 = LinearRegression().fit(x2.values.reshape(-1, 1), y2)

        r2_1, r2_2 = r2_score(y1, model1.predict(x1.values.reshape(-1, 1))), r2_score(
            y2, model2.predict(x2.values.reshape(-1, 1))
        )
        slope1, _, n1 = model1.coef_[0], model1.intercept_, len(x1)
        y_pred1 = model1.predict(x1.values.reshape(-1, 1))
        se_slope1 = np.sqrt(np.sum((y1 - y_pred1) ** 2) / (n1 - 2)) / np.sqrt(np.sum((x1 - np.mean(x1)) ** 2))
        t_stat1 = slope1 / se_slope1
        p_value1 = 2 * (1 - stats.t.cdf(np.abs(t_stat1), df=n1 - 2))

        slope2, _, n2 = model2.coef_[0], model2.intercept_, len(x2)
        y_pred2 = model2.predict(x2.values.reshape(-1, 1))
        se_slope2 = np.sqrt(np.sum((y2 - y_pred2) ** 2) / (n2 - 2)) / np.sqrt(np.sum((x2 - np.mean(x2)) ** 2))
        t_stat2 = slope2 / se_slope2
        p_value2 = 2 * (1 - stats.t.cdf(np.abs(t_stat2), df=n2 - 2))

        plt.subplot(3, 5, i + 1)
        sns.regplot(
            x=x1,
            y=y1,
            color="blue",
            label=f"Data1 (R²={r2_1:.2f}, p={p_value1:.3f})",
            scatter_kws={"s": 50},
            line_kws={"linestyle": "--"},
            ci=None,
        )
        sns.regplot(
            x=x2,
            y=y2,
            color="red",
            label=f"Data2 (R²={r2_2:.2f}, p={p_value2:.3f})",
            scatter_kws={"s": 50},
            line_kws={"linestyle": "--"},
            ci=None,
        )
        cat1_title = cat1.replace(" weighted_score", "").replace(" Success", "").replace("coef", "").replace(" raw", "")
        cat2_title = cat2.replace(" weighted_score", "").replace(" Success", "").replace("coef", "").replace(" raw", "")
        plt.title(f"{cat1_title} vs {cat2_title}")
        plt.xlabel(cat1_title)
        plt.ylabel(cat2_title)
        plt.legend()

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def process_data(df, nasa_categories, columns_of_interest, output_dir, term):
    """Process the data to calculate NASA TLX raw and weighted scores and coefficients."""

    for column in nasa_categories:
        df[f"{column} raw"] = df[column].apply(convert_text_to_score)
        df[f"{column} count"] = count_instances(df, column, columns_of_interest)
        df[f"{column} coef"] = df[f"{column} count"] / 15
        df[f"{column} weighted_score"] = df[f"{column} raw"] * df[f"{column} coef"]

    df["Weighted Score"] = calculate_overall_nasa_tlx_weighted_score(df)
    df["Raw Score"] = df[[f"{col} raw" for col in nasa_categories]].mean(axis=1)
    df.to_csv(output_dir + f"/nasa_tlx_scores_{term}.csv", index=False)
    return df


def format_user_list(user_list):
    """Add one to each user in a given list to avoid 'user 0'."""
    return [user + 1 for user in user_list]


def categorize_users(with_scores, without_scores):
    """Categorize users into three groups based on their change in task load (NASA TLX weighted scores)."""
    decrease_users = []
    increase_users = []
    no_change_users = []

    for i in range(len(with_scores)):
        if with_scores[i] < without_scores[i]:
            decrease_users.append(i)
        elif with_scores[i] > without_scores[i]:
            increase_users.append(i)
        else:
            no_change_users.append(i)

    return decrease_users, increase_users, no_change_users


def perform_stat_tests_for_groups(with_scores, without_scores, user_groups):
    """Perform paired t-test and Wilcoxon signed-rank test on the selected users."""
    p_values = {}

    for group_name, users in user_groups.items():
        if users:  # Check if the list is not empty
            p_values[group_name] = perform_stat_tests(with_scores, without_scores, users)

    return p_values


def categorize_and_perform_tests(with_scores, without_scores, user_groups, output_dir):
    """Categorize users and perform statistical tests for each group."""
    p_values = {}

    for group_name, users in user_groups.items():
        p_values[group_name] = {
            "weighted": (
                perform_stat_tests(with_scores["weighted"], without_scores["weighted"], users["weighted"])
                if users["weighted"]
                else (None, None)
            ),
            "raw": (
                perform_stat_tests(with_scores["raw"], without_scores["raw"], users["raw"])
                if users["raw"]
                else (None, None)
            ),
        }

    # Write the stats results to a file
    write_stats_to_file(output_dir, user_groups, p_values)


def write_stats_to_file(output_dir, user_groups, p_values):
    """Write paired t-test and Wilcoxon statistical test results to a file, given a user group."""

    def format_p_value(p_value):
        """Format the p-value to 5 decimal places."""
        return f"{p_value:.5f}" if p_value is not None else "N/A"

    with open(output_dir + "/users_mod_task_load_signif_tests.txt", "w") as file:
        file.write(
            "Did Copilot use significantly impact task load for users in decreased, increased, or no change groups?\n"
        )

        for group_name, users in user_groups.items():
            if users["weighted"] or users["raw"]:  # Check if the list is not empty
                if group_name == "all users":
                    file.write(f"\nAll users:")
                else:
                    file.write(f"\nUsers who experienced a {group_name} in NASA task load: ")
                file.write(
                    f"(users by NASA TLX weighted: {format_user_list(users['weighted'])}, "
                    f"and raw: {format_user_list(users['raw'])} scores):\n"
                )
                file.write(
                    f"- Paired t-test p-value for weighted scores: "
                    f"{format_p_value(p_values[group_name]['weighted'][0])}\n"
                )
                file.write(
                    (
                        f"- Wilcoxon signed-rank test p-value for weighted scores: "
                        f"{format_p_value(p_values[group_name]['weighted'][1])}\n"
                    )
                )
                file.write(
                    f"- Paired t-test p-value for raw scores: {format_p_value(p_values[group_name]['raw'][0])}\n"
                )
                file.write(
                    f"- Wilcoxon signed-rank test p-value for raw scores: "
                    f"{format_p_value(p_values[group_name]['raw'][1])}\n"
                )


def plot_avg_nasa_tlx_per_cat(without_copilot, with_copilot, nasa_categories, output_dir):
    """
    Create and save a bar plot comparing the average NASA TLX raw and weighted scores
    for each category, without and with Copilot.
    """
    without_copilot["Condition"] = "Without_Copilot"
    with_copilot["Condition"] = "With_Copilot"
    without_copilot["User"] = range(1, len(without_copilot) + 1)
    with_copilot["User"] = range(1, len(with_copilot) + 1)

    plot_params = [
        (
            "raw",
            "Average Raw NASA TLX Scores by Category Without and With Copilot",
            "Average Score",
            "average_raw_scores.png",
        ),
        (
            "weighted_score",
            "Average Weighted NASA TLX Scores by Category Without and With Copilot",
            "Average Weighted Score",
            "average_weighted_scores.png",
        ),
        (
            "coef",
            "Average NASA TLX Coefficients by Category Without and With Copilot",
            "Average Coefficient",
            "average_coefficients.png",
        ),
    ]

    for suffix, title, ylabel, filename in plot_params:
        columns = [f"{col} {suffix}" for col in nasa_categories]
        df1 = without_copilot[columns + ["User", "Condition"]].copy()
        df2 = with_copilot[columns + ["User", "Condition"]].copy()
        plot_average_scores(df1, df2, columns, title, ylabel, filename, output_dir)


def create_combined_scatter_plots(without_copilot, with_copilot, nasa_categories, output_dir):
    """Create scatter plots to compare the NASA TLX raw and weighted scores, and the NASA TLX coefficients."""
    plot_params = [
        ("raw", "Combined Scatter Plots for Raw Scores", "combined_scatter_plots_raw.png"),
        ("weighted_score", "Combined Scatter Plots for Weighted Scores", "combined_scatter_plots_weighted.png"),
        ("coef", "Combined Scatter Plots for Coefficients", "combined_scatter_plots_coef.png"),
    ]

    for suffix, title, filename in plot_params:
        columns = [f"{col} {suffix}" for col in nasa_categories]
        create_combined_scatter_plot(without_copilot, with_copilot, columns, title, filename, output_dir)


def main(args):
    # Set up plot characteristics
    plt.rcParams.update({"font.size": 14})
    users = np.arange(1, args.num_users)
    bar_width = args.bar_width

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.outdir, f"nasa_tlx_plots_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Load user study data
    without_copilot = pd.read_csv(args.without_copilot_table)
    with_copilot = pd.read_csv(args.with_copilot_table)

    # 1. Calculate and plot the NASA TLX weighted and raw scores for each user, without and with the Copilot

    # NASA TLX categories
    nasa_categories = [
        "Mental Demand",
        "Physical Demand",
        "Time Pressure",
        "Personal Performance Success",
        "Effort Level",
        "Frustration Level",
    ]

    # Columns to determine frequency/count per category
    prefix = "Which factor was the more important contributor to the workload for rare disease case analysis?"
    columns_to_count = [col for col in without_copilot.columns if col.startswith(prefix)]

    without_copilot = process_data(without_copilot, nasa_categories, columns_to_count, output_dir, "without_copilot")
    with_copilot = process_data(with_copilot, nasa_categories, columns_to_count, output_dir, "with_copilot")

    # Gather weighted scores for each user into a list for processing
    without_weighted_scores = without_copilot["Weighted Score"].tolist()
    with_weighted_scores = with_copilot["Weighted Score"].tolist()

    # Gather raw scores for each user into a list for processing
    without_raw_scores = without_copilot["Raw Score"].tolist()
    with_raw_scores = with_copilot["Raw Score"].tolist()

    # Create and save bar plots comparing NASA TLX weighted scores for each user without and with Copilot
    create_bar_plot(
        users,
        without_weighted_scores,
        with_weighted_scores,
        "Weighted Score",
        "Weighted NASA TLX Scores: without v.s. with Copilot",
        "weighted_scores_by_user.png",
        bar_width,
        output_dir,
    )

    # Create and save bar plots comparing NASA TLX raw scores for each user without and with Copilot
    create_bar_plot(
        users,
        without_raw_scores,
        with_raw_scores,
        "Raw Score",
        "Raw NASA TLX Scores: without v.s. with Copilot",
        "raw_scores_by_user.png",
        bar_width,
        output_dir,
    )

    # 2. Calculate and plot the distribution of percent changes in the weighted and raw NASA TLX scores,
    #    to illustrate that these are likely not normally distributed and rule out the paired t-test.

    # Calculate percent changes in NASA TLX weighted scores
    percent_changes_weighted = [
        ((with_weighted_scores[i] - without_weighted_scores[i]) / without_weighted_scores[i]) * 100
        for i in range(len(users))
    ]

    # Calculate percent changes in NASA TLX weighted scores
    percent_changes_raw = [
        ((with_raw_scores[i] - without_raw_scores[i]) / without_raw_scores[i]) * 100 for i in range(len(users))
    ]

    # Plot the distribution of percent changes in those NASA TLX scores
    plot_distribution(
        percent_changes_weighted,
        "Distribution of Percent Changes in NASA TLX Weighted Scores",
        "percent_changes_weighted.png",
        "blue",
        output_dir,
    )
    plot_distribution(
        percent_changes_raw,
        "Distribution of Percent Changes in NASA TLX Raw Scores",
        "percent_changes_raw.png",
        "green",
        output_dir,
    )

    # 3. Stastical tests are run to identify any significant relationships between the users for which there was a
    #    decrease in task load compared to those who experienced an increase.

    # Convert the user list to 0-based indexing
    users_0_based = list(users - 1)

    # Categorize users into groups based on their change in task load
    decrease_users_w, increase_users_w, no_change_users_w = categorize_users(
        with_weighted_scores, without_weighted_scores
    )
    decrease_users_r, increase_users_r, no_change_users_r = categorize_users(with_raw_scores, without_raw_scores)

    user_groups = {
        "all users": {"weighted": users_0_based, "raw": users_0_based},  # All users
        "decrease": {"weighted": decrease_users_w, "raw": decrease_users_r},
        "increase": {"weighted": increase_users_w, "raw": increase_users_r},
        "no change": {"weighted": no_change_users_w, "raw": no_change_users_r},
    }

    # Perform statistical tests for each group
    with_scores = {"weighted": with_weighted_scores, "raw": with_raw_scores}
    without_scores = {"weighted": without_weighted_scores, "raw": without_raw_scores}
    categorize_and_perform_tests(with_scores, without_scores, user_groups, output_dir)

    # 4. Calculate and plot the average NASA TLX raw and weighted scores for each of 6 categories, without and with
    #    the Copilot to identify any potential significant changes in task load due to Copilot use.

    plot_avg_nasa_tlx_per_cat(without_copilot, with_copilot, nasa_categories, output_dir)

    # 5. Create scatter plots to compare the NASA TLX raw scores, weighted scores, and coefficients for users without
    #    and with Copilot for each pair of categories, to identify any potential relationships between the categories.

    create_combined_scatter_plots(without_copilot, with_copilot, nasa_categories, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NASA TLX calculator.")
    parser.add_argument(
        "--without_copilot_table",
        type=str,
        default=(r"data/user_study/Evidence_Aggregator_User_Acceptance_Study_Day_1_Session_Data(1_8).csv"),
        help=(
            "User study data without Copilot "
            "(i.e. Evidence_Aggregator_User_Acceptance_Study_Day_1_Session_Data(1_8).csv)"
        ),
    )
    parser.add_argument(
        "--with_copilot_table",
        type=str,
        default=(r"data/user_study/Evidence_Aggregator_User_Acceptance_Study_Day_2_Session_Data(1_8).csv"),
        help=(
            "User study data with Copilot "
            "(i.e. Evidence_Aggregator_User_Acceptance_Study_Day_2_Session_Data(1_8).csv)"
        ),
    )
    parser.add_argument("--bar-width", default=0.35, type=float, help="the width of the bars in the bar plots")
    parser.add_argument("--num-users", default=9, type=int)
    parser.add_argument("--outdir", default=".out/", type=str)
    args = parser.parse_args()

    main(args)
