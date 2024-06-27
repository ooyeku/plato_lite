import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from scipy import stats
from utils.logger import PlatoLogger

logger = PlatoLogger().logger


class Insights:
    def __init__(self, df):
        self.df = df.copy()

    # Qualitative Analysis Methods
    def sentiment_analysis(self, text_column):
        sid = SentimentIntensityAnalyzer()
        self.df['sentiment'] = self.df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
        logger.info(f"Sentiment analysis performed on column: {text_column}")
        return self.df

    def generate_wordcloud(self, text_column, max_words=100, background_color="white"):
        text = ' '.join(self.df[text_column].dropna())
        wordcloud = WordCloud(max_words=max_words, background_color=background_color).generate(text)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        logger.info(f"Word cloud generated for column: {text_column}")

    def keyword_extraction(self, text_column, method='tfidf', top_n=10):
        vectorizer = TfidfVectorizer(stop_words='english') if method == 'tfidf' else CountVectorizer(
            stop_words='english')
        X = vectorizer.fit_transform(self.df[text_column].dropna())
        scores = np.sum(X.toarray(), axis=0)
        keywords = vectorizer.get_feature_names_out()
        keyword_scores = pd.DataFrame({'keyword': keywords, 'score': scores}).sort_values(by='score',
                                                                                          ascending=False).head(top_n)
        logger.info(f"Keyword extraction performed on column: {text_column} using {method} method")
        return keyword_scores

    # Quantitative Analysis Methods
    def descriptive_statistics(self):
        desc_stats = self.df.describe()
        logger.info("Descriptive statistics calculated")
        return desc_stats

    def correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        logger.info("Correlation matrix calculated")
        return corr_matrix

    def hypothesis_testing(self, column1, column2, test='t-test'):
        if test == 't-test':
            statistic, p_value = stats.ttest_ind(self.df[column1], self.df[column2], nan_policy='omit')
        elif test == 'anova':
            statistic, p_value = stats.f_oneway(self.df[column1], self.df[column2])
        else:
            raise ValueError(f"Unknown test: {test}")
        logger.info(f"Hypothesis testing performed: {test} between {column1} and {column2}")
        return {'statistic': statistic, 'p_value': p_value}

    # Modeling Methods
    def train_test_split(self, target, features, test_size=0.2, random_state=42):
        X = self.df[features]
        y = self.df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, X_test, y_train, y_test, model_type, params=None):
        models = {
            'linear': LinearRegression(),
            'logistic': LogisticRegression(max_iter=1000),
            'dt_classifier': DecisionTreeClassifier(),
            'dt_regressor': DecisionTreeRegressor(),
            'rf_classifier': RandomForestClassifier(),
            'rf_regressor': RandomForestRegressor()
        }

        model = models.get(model_type)
        if not model:
            raise ValueError(f"Unknown model type: {model_type}")

        if params:
            model = GridSearchCV(model, params, cv=5)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if model_type in ['linear', 'dt_regressor', 'rf_regressor']:
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            return {'model': model, 'predictions': predictions, 'mse': mse, 'r2': r2}
        else:
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            return {'model': model, 'predictions': predictions, 'accuracy': accuracy, 'report': report}

    # Visualization Methods
    def plot(self, plot_type, **kwargs):
        plt.figure(figsize=(10, 6))

        if plot_type == 'histogram':
            sns.histplot(self.df[kwargs['column']], bins=kwargs.get('bins', 10), kde=True)
        elif plot_type == 'scatter':
            sns.scatterplot(x=self.df[kwargs['x']], y=self.df[kwargs['y']])
        elif plot_type == 'box':
            sns.boxplot(x=self.df[kwargs['x']], y=self.df[kwargs['y']])
        elif plot_type == 'heatmap':
            sns.heatmap(self.correlation_matrix(), annot=True, cmap='coolwarm')
        elif plot_type == 'line':
            sns.lineplot(x=self.df[kwargs['x']], y=self.df[kwargs['y']])
        elif plot_type == 'bar':
            sns.barplot(x=self.df[kwargs['x']], y=self.df[kwargs['y']])
        elif plot_type == 'violin':
            sns.violinplot(x=self.df[kwargs['x']], y=self.df[kwargs['y']])
        elif plot_type == 'pairplot':
            sns.pairplot(self.df, hue=kwargs.get('hue'))
        elif plot_type == 'distribution':
            sns.displot(self.df[kwargs['column']])
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        plt.title(kwargs.get('title', f'{plot_type.capitalize()} Plot'))
        plt.xlabel(kwargs.get('xlabel', kwargs.get('x', '')))
        plt.ylabel(kwargs.get('ylabel', kwargs.get('y', '')))
        plt.show()

    def plot_3d_scatter(self, x, y, z, color=None, title=None):
        fig = px.scatter_3d(self.df, x=x, y=y, z=z, color=color, title=title)
        fig.show()

    def plot_facet_grid(self, row, col, x, y, hue=None):
        g = sns.FacetGrid(self.df, row=row, col=col, hue=hue)
        g.map(sns.scatterplot, x, y)
        plt.show()



    def plot_scatter(self, x, y, **kwargs):
        """Create a scatter plot from data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=self.df[x], y=self.df[y], ax=ax)
        ax.set_title(kwargs.get('title', f'Scatter Plot of {x} vs {y}'))
        ax.set_xlabel(kwargs.get('xlabel', x))
        ax.set_ylabel(kwargs.get('ylabel', y))
        plt.show()
        logger.info(f"Created scatter plot of {x} vs {y}")
        return fig

# Example usage
if __name__ == "__main__":
    # Load your data
    data = {
        'text': ['I love this product', 'This is terrible', 'Neutral opinion'],
        'score': [5, 1, 3],
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'category': ['A', 'B', 'A']
    }
    df = pd.DataFrame(data)

    # Initialize Insights
    insights = Insights(df)

    # Qualitative analysis
    insights.sentiment_analysis('text')
    insights.generate_wordcloud('text')
    keywords = insights.keyword_extraction('text')
    print("Top keywords:", keywords)

    # Quantitative analysis
    print(insights.descriptive_statistics())
    print(insights.correlation_matrix())
    print(insights.hypothesis_testing('score', 'age'))

    # Modeling
    X_train, X_test, y_train, y_test = insights.train_test_split('score', ['age', 'income'])
    model_results = insights.train_model(X_train, X_test, y_train, y_test, 'linear')
    print("Model R2 score:", model_results['r2'])

    # Visualization
    insights.plot('histogram', column='age')
    insights.plot('scatter', x='age', y='income')
    insights.plot('heatmap')
    insights.plot_3d_scatter('age', 'income', 'score', color='category')
    insights.plot_facet_grid('category', 'age', 'score', 'income')