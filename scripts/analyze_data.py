import os
import pandas as pd


NUMERIC_COLS = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Addicted_Score",
    "Age",
    "Conflicts_Over_Social_Media",
]

class DataAnalyzer:
    def __init__(self, csv_path, output_dir="results"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = pd.read_csv(csv_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def numeric_summary_by_group(self, group_col, sort_by_count=False):
        def q25(x):
            return x.quantile(0.25)

        def q75(x):
            return x.quantile(0.75)

        grouped = (
            self.df.groupby(group_col)[NUMERIC_COLS]
            .agg(["mean", "std", "min", "max", "median", q25, q75])
        )
        if sort_by_count:
            order = self.df[group_col].value_counts(dropna=False).index
            return grouped.reindex(order)
        return grouped.sort_index()

    def categorical_counts(self, col):
        return self.df[col].value_counts(dropna=False).to_frame(name="count")

    def correlation_matrix(self, method="pearson"):
        return self.df[NUMERIC_COLS].corr(method=method)

    def save_correlation_heatmap(self, corr, path, method_label):
        # Local imports keep plotting optional for data-only runs.
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        labels = {
            "Age": "Age",
            "Avg_Daily_Usage_Hours": "Avg Daily\nUsage (hrs)",
            "Sleep_Hours_Per_Night": "Sleep\nHours/Night",
            "Mental_Health_Score": "Mental Health\nScore",
            "Conflicts_Over_Social_Media": "Conflicts\nOver Social Media",
            "Addicted_Score": "Addiction\nScore",
        }
        display_labels = [labels.get(col, col) for col in corr.columns]
        corr = corr.copy()
        for i in range(len(corr)):
            corr.iat[i, i] = 1.0
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_facecolor("#e6e6e6")
        sns.heatmap(
            corr,
            mask=mask,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.6,
            square=True,
            cbar_kws={"label": method_label, "shrink": 0.85},
            ax=ax,
        )
        ax.set_title(f"Correlation Matrix (Numeric Variables, {method_label})")
        ax.set_xticklabels(display_labels, rotation=45, ha="right")
        ax.set_yticklabels(display_labels, rotation=0)
        ax.tick_params(axis="both", labelsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    @staticmethod
    def df_to_markdown(table):
        """transform an dataframe into table in markdown file"""
        table = table.copy()
        table = table.round(3)

        def fmt(val):
            text = "" if pd.isna(val) else str(val)
            return text.replace("|", "\\|")

        index_name = table.index.name if table.index.name is not None else ""
        header = [index_name] + [str(c) for c in table.columns]
        rows = []
        for idx, row in table.iterrows():
            rows.append([str(idx)] + [fmt(v) for v in row.tolist()])

        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def build_tables(self):
        tables = {}

        tables["overall_numeric_summary"] = self.df[NUMERIC_COLS].describe().T
        tables["age_numeric_summary"] = self.numeric_summary_by_group("Age")

        tables["gender_counts"] = self.categorical_counts("Gender")
        tables["gender_numeric_summary"] = self.numeric_summary_by_group("Gender")

        tables["academic_level_counts"] = self.categorical_counts("Academic_Level")
        tables["academic_level_numeric_summary"] = self.numeric_summary_by_group(
            "Academic_Level", sort_by_count=True
        )

        level_means = self.df.groupby("Academic_Level")[NUMERIC_COLS].mean()
        if "High School" in level_means.index:
            tables["academic_level_vs_high_school_deltas"] = level_means.sub(
                level_means.loc["High School"]
            )

        tables["country_counts"] = self.categorical_counts("Country")
        tables["country_numeric_summary"] = self.numeric_summary_by_group(
            "Country", sort_by_count=True
        )

        tables["most_used_platform_counts"] = self.categorical_counts(
            "Most_Used_Platform"
        )
        tables["most_used_platform_numeric_summary"] = self.numeric_summary_by_group(
            "Most_Used_Platform", sort_by_count=True
        )

        tables["relationship_status_counts"] = self.categorical_counts(
            "Relationship_Status"
        )
        tables["relationship_status_numeric_summary"] = self.numeric_summary_by_group(
            "Relationship_Status"
        )

        tables["affects_academic_performance_counts"] = self.categorical_counts(
            "Affects_Academic_Performance"
        )
        tables["affects_academic_performance_numeric_summary"] = (
            self.numeric_summary_by_group("Affects_Academic_Performance")
        )

        tables["numeric_correlation_matrix"] = self.correlation_matrix("pearson")
        tables["numeric_correlation_matrix_spearman"] = self.correlation_matrix(
            "spearman"
        )

        return tables

    def write_outputs(self, tables):
        for name, table in tables.items():
            table.round(3).to_csv(os.path.join(self.output_dir, f"{name}.csv"))

        markdown_path = os.path.join(self.output_dir, "summary_tables.md")
        with open(markdown_path, "w", encoding="utf-8") as f:
            for name, table in tables.items():
                f.write(f"## {name}\n\n")
                f.write(self.df_to_markdown(table))
                f.write("\n\n")

        try:
            with pd.ExcelWriter(
                os.path.join(self.output_dir, "summary_tables.xlsx")
            ) as writer:
                for name, table in tables.items():
                    table.round(3).to_excel(writer, sheet_name=name[:31])
        except Exception:
            pass

        heatmap_path = os.path.join(
            self.output_dir, "numeric_correlation_heatmap_pearson.png"
        )
        self.save_correlation_heatmap(
            tables["numeric_correlation_matrix"],
            heatmap_path,
            "Pearson r",
        )

        heatmap_path = os.path.join(
            self.output_dir, "numeric_correlation_heatmap_spearman.png"
        )
        self.save_correlation_heatmap(
            tables["numeric_correlation_matrix_spearman"],
            heatmap_path,
            "Spearman rho",
        )


def main():
    analyzer = DataAnalyzer("Data/media_addiction.csv")
    tables = analyzer.build_tables()
    analyzer.write_outputs(tables)


if __name__ == "__main__":
    main()
