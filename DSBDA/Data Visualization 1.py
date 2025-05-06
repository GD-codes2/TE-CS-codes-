import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load dataset
titanic = sns.load_dataset('titanic')

def plot_distributions():
    """Plot distributions of key numerical features"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Fare distribution
    sns.histplot(titanic['fare'], bins=30, kde=True, ax=ax1)
    ax1.set_title("Ticket Fare Distribution")
    ax1.set_xlabel("Fare ($)")
    ax1.set_ylabel("Frequency")
    
    # Age distribution
    sns.histplot(titanic['age'].dropna(), bins=20, kde=True, ax=ax2)
    ax2.set_title("Age Distribution of Passengers")
    ax2.set_xlabel("Age (years)")
    ax2.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def plot_age_fare_relationships():
    """Visualize relationships between age and fare"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    sns.jointplot(x='age', y='fare', data=titanic, kind='scatter')
    plt.suptitle("Age vs Fare (Scatter Plot)")
    
    # Hex plot
    sns.jointplot(x='age', y='fare', data=titanic, kind='hex')
    plt.suptitle("Age vs Fare (Hexbin Plot)")
    
    plt.tight_layout()
    plt.show()

def plot_gender_analysis():
    """Analyze gender distribution and age patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gender count
    sns.countplot(x='sex', data=titanic, ax=axes[0, 0])
    axes[0, 0].set_title("Passenger Count by Gender")
    
    # Age by gender
    sns.boxplot(x='sex', y='age', data=titanic, ax=axes[0, 1])
    axes[0, 1].set_title("Age Distribution by Gender")
    
    # Age by gender and survival
    sns.boxplot(x='sex', y='age', data=titanic, hue='survived', ax=axes[1, 0])
    axes[1, 0].set_title("Age by Gender and Survival Status")
    
    # Violin plot
    sns.violinplot(x='sex', y='age', data=titanic, ax=axes[1, 1])
    axes[1, 1].set_title("Age Distribution by Gender (Violin Plot)")
    
    plt.tight_layout()
    plt.show()

def plot_correlation_analysis():
    """Analyze correlations between numerical features"""
    plt.figure(figsize=(12, 8))
    
    # Select only numerical columns
    numeric_data = titanic.select_dtypes(include=['number'])
    
    # Correlation heatmap
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Titanic Dataset")
    plt.tight_layout()
    plt.show()
    
    # Cluster map
    sns.clustermap(numeric_data.corr(), cmap="coolwarm", annot=True)
    plt.title("Cluster Map of Titanic Dataset")
    plt.tight_layout()
    plt.show()

def main():
    plot_distributions()
    plot_age_fare_relationships()
    plot_gender_analysis()
    plot_correlation_analysis()

if __name__ == "__main__":
    main()