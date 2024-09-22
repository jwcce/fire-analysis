import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class FireDataProcessor:
    def __init__(self, data_source):
        """Initialize with either a file path or a DataFrame."""
        if isinstance(data_source, str):
            # If data_source is a file path
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            # If data_source is a pandas DataFrame
            self.df = data_source
        else:
            raise ValueError("Invalid data source. Must be a file path or a DataFrame.")

        self._prepare_data()

    def _prepare_data(self):
        """Format the dataset for easier analysis."""
        self.df['Year'] = pd.to_datetime(self.df['Year'], format='%Y')
        self.df['Month'] = pd.to_datetime(self.df['Month'], format='%B')
        self.df['State'] = self.df['State'].str.title()

    def get_fires_per_year(self):
        """Returns the total number of fires per year."""
        return self.df.groupby(self.df['Year'].dt.year)['Number of Fires'].sum().reset_index()

    def get_fires_per_state(self):
        """Returns the total number of fires per state."""
        return self.df.groupby('State')['Number of Fires'].sum().reset_index()

    def get_fires_per_month(self):
        """Returns the total number of fires per month."""
        return self.df.groupby(self.df['Month'].dt.month)['Number of Fires'].sum().reset_index()


class FireVisualizer:
    """Handles visualization of fire data."""
    
    def __init__(self, fires_per_year, fires_per_state, fires_per_month):
        self.fires_per_year = fires_per_year
        self.fires_per_state = fires_per_state
        self.fires_per_month = fires_per_month
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    def plot_fires_per_year(self):
        self.axs[0, 0].plot(self.fires_per_year['Year'], self.fires_per_year['Number of Fires'], marker='o')
        self.axs[0, 0].set_title('Total Number of Fires per Year')
        self.axs[0, 0].set_xlabel('Year')
        self.axs[0, 0].set_ylabel('Number of Fires')
        self.axs[0, 0].grid(True)

    def plot_fires_per_state(self):
        self.axs[0, 1].bar(self.fires_per_state['State'], self.fires_per_state['Number of Fires'])
        self.axs[0, 1].set_title('Total Number of Fires per State')
        self.axs[0, 1].set_xlabel('State')
        self.axs[0, 1].set_ylabel('Number of Fires')
        self.axs[0, 1].tick_params(axis='x', rotation=90)

    def plot_fires_per_month(self):
        self.axs[1, 0].plot(self.fires_per_month['Month'], self.fires_per_month['Number of Fires'], marker='o')
        self.axs[1, 0].set_title('Total Number of Fires per Month')
        self.axs[1, 0].set_xlabel('Month')
        self.axs[1, 0].set_ylabel('Number of Fires')
        self.axs[1, 0].grid(True)

    def show(self):
        self.fig.tight_layout()
        plt.show()


class FirePredictor:
    """Handles prediction of future fires using polynomial regression."""
    
    def __init__(self, fires_per_year):
        self.fires_per_year = fires_per_year
        self.model = None
        self.poly_features = PolynomialFeatures(degree=3)
        self._train_model()

    def _train_model(self):
        """Train a polynomial regression model."""
        X = self.fires_per_year[['Year']]
        y = self.fires_per_year['Number of Fires']
        X_poly = self.poly_features.fit_transform(X)
        self.model = LinearRegression()
        self.model.fit(X_poly, y)

    def predict_fires(self, future_years):
        """Predict fires for future years."""
        future_years_poly = self.poly_features.transform(future_years.reshape(-1, 1))
        return self.model.predict(future_years_poly)

    def plot_predictions(self, ax):
        """Plot predictions alongside actual data."""
        X = self.fires_per_year[['Year']]
        y = self.fires_per_year['Number of Fires']
        future_years = np.array([2024, 2025, 2026, 2027, 2028])
        
        future_fires = self.predict_fires(future_years)
        ax.plot(self.fires_per_year['Year'], y, marker='o', label='Actual')
        ax.plot(np.concatenate((self.fires_per_year['Year'], future_years)), 
                np.concatenate((y, future_fires)), marker='o', linestyle='--', label='Predicted')
        ax.scatter(future_years, future_fires, marker='o', label='Future Predictions')

        ax.set_title('Predicted Total Number of Fires per Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Fires')
        ax.grid(True)
        ax.legend()


def main():
    # File path to the dataset
    file_path = 'brazilian-fire-dataset.csv'

    # Process the data
    processor = FireDataProcessor(file_path)
    fires_per_year = processor.get_fires_per_year()
    fires_per_state = processor.get_fires_per_state()
    fires_per_month = processor.get_fires_per_month()

    # Visualize the data
    visualizer = FireVisualizer(fires_per_year, fires_per_state, fires_per_month)
    visualizer.plot_fires_per_year()
    visualizer.plot_fires_per_state()
    visualizer.plot_fires_per_month()

    # Predict future fires
    predictor = FirePredictor(fires_per_year)
    predictor.plot_predictions(visualizer.axs[1, 1])

    # Show the plots
    visualizer.show()


if __name__ == '__main__':
    main()
