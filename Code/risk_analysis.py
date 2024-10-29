import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq

class RiskAnalysis:
    def __init__(self, df):
        self.df = df

    def plot_pd_distribution(self):
        """Distribution of Probability of Default (PD)"""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df["Prob_default"], bins=30, kde=True, color='purple')
        plt.title("Distribution of Probability of Default (PD)")
        plt.xlabel("Probability of Default")
        plt.ylabel("Frequency")
        plt.show()

    def plot_risk_by_category(self, category):
        """Risk Assessment by Category (e.g., home ownership)"""
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=category, y="Risk Assessment", data=self.df)
        plt.title(f"Risk Assessment by {category}")
        plt.xlabel(category)
        plt.ylabel("Risk Assessment")
        plt.show()

    def plot_risk_concentration_curve(self):
        """Risk Concentration Curve (Lorenz Curve)"""
        sorted_risk = np.sort(self.df["Risk Assessment"].values)
        cum_risk = np.cumsum(sorted_risk) / np.sum(sorted_risk)
        cum_count = np.arange(1, len(sorted_risk) + 1) / len(sorted_risk)

        plt.figure(figsize=(10, 6))
        plt.plot(cum_count, cum_risk, label="Lorenz Curve", color="blue")
        plt.plot([0, 1], [0, 1], '--', color="grey", label="Perfect Distribution")
        plt.xlabel("Proportion of Applicants")
        plt.ylabel("Proportion of Accumulated Risk")
        plt.legend()
        plt.title("Risk Concentration Curve")
        plt.show()

    def plot_risk_by_income_range(self, bins=5):
        """Risk by Income Range Interval"""
        self.df['income_range'] = pd.cut(self.df['person_income'], bins=bins)
        income_risk = self.df.groupby('income_range')['Risk Assessment'].mean()
        
        plt.figure(figsize=(10, 6))
        income_risk.plot(kind='bar', color='teal')
        plt.title("Average Risk by Income Range Interval")
        plt.xlabel("Income Range")
        plt.ylabel("Average Risk Assessment")
        plt.xticks(rotation=45)
        plt.show()

    def calculate_total_portfolio_risk(self):
        """Calculate Total Portfolio Risk"""
        total_risk = self.df["Risk Assessment"].sum()
        print(f"Total Portfolio Risk: ${total_risk:,.2f}")