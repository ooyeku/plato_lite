# utils/data_generator.py

import numpy as np
import pandas as pd
from faker import Faker
from typing import Dict, Any, List, Union
from datetime import datetime, timedelta
import random
import string

class DataGenerator:
    """
    A flexible and powerful data generator for creating synthetic datasets.

    This class provides an easy-to-use interface for generating various types of data,
    including numerical, categorical, temporal, and text data. It supports custom
    distributions and relationships between columns.

    Attributes:
        num_rows (int): The number of rows to generate in the dataset.
        seed (int): Random seed for reproducibility.
        fake (Faker): Faker instance for generating realistic fake data.
        data (pd.DataFrame): The generated dataset.

    Example:
        generator = DataGenerator(1000)
        generator.add_numerical("age", min_value=18, max_value=80, distribution="normal")
        generator.add_categorical("gender", categories=["Male", "Female", "Other"], weights=[0.48, 0.48, 0.04])
        generator.add_datetime("registration_date", start_date="2020-01-01", end_date="2023-06-30")
        generator.add_text("comment", min_words=5, max_words=20)
        generator.add_email("email")
        generator.add_phone_number("phone")
        generator.add_dependent("salary", depends_on="age", func=lambda age: age * 1000 + np.random.normal(0, 5000))
        df = generator.generate()
    """

    def __init__(self, num_rows: int, seed: int = None):
        self.num_rows = num_rows
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
        self.data = pd.DataFrame(index=range(num_rows))
        self._column_generators = {}

    def add_numerical(self, name: str, min_value: float, max_value: float, distribution: str = "uniform", **kwargs):
        """Add a numerical column with specified distribution."""
        if distribution == "uniform":
            self._column_generators[name] = lambda: np.random.uniform(min_value, max_value, self.num_rows)
        elif distribution == "normal":
            mean = kwargs.get("mean", (min_value + max_value) / 2)
            std = kwargs.get("std", (max_value - min_value) / 6)
            self._column_generators[name] = lambda: np.clip(np.random.normal(mean, std, self.num_rows), min_value, max_value)
        elif distribution == "exponential":
            scale = kwargs.get("scale", 1.0)
            self._column_generators[name] = lambda: np.clip(np.random.exponential(scale, self.num_rows), min_value, max_value)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def add_categorical(self, name: str, categories: List[str], weights: List[float] = None):
        """Add a categorical column with optional weights."""
        self._column_generators[name] = lambda: np.random.choice(categories, self.num_rows, p=weights)

    def add_datetime(self, name: str, start_date: str, end_date: str, format: str = "%Y-%m-%d"):
        """Add a datetime column within the specified range."""
        start = datetime.strptime(start_date, format)
        end = datetime.strptime(end_date, format)
        delta = end - start
        self._column_generators[name] = lambda: [start + timedelta(days=random.random() * delta.days) for _ in range(self.num_rows)]

    def add_text(self, name: str, min_words: int = 5, max_words: int = 15):
        """Add a text column with a specified word range."""
        self._column_generators[name] = lambda: [self.fake.sentence(nb_words=random.randint(min_words, max_words)) for _ in range(self.num_rows)]

    def add_email(self, name: str):
        """Add an email column."""
        self._column_generators[name] = lambda: [self.fake.email() for _ in range(self.num_rows)]

    def add_phone_number(self, name: str):
        """Add a phone number column."""
        self._column_generators[name] = lambda: [self.fake.phone_number() for _ in range(self.num_rows)]

    def add_dependent(self, name: str, depends_on: str, func: callable):
        """Add a column that depends on another column."""
        self._column_generators[name] = lambda: func(self.data[depends_on])

    def add_custom(self, name: str, generator: callable):
        """Add a column with a custom generator function."""
        self._column_generators[name] = generator

    def generate(self) -> pd.DataFrame:
        """Generate the dataset based on the defined columns."""
        for name, generator in self._column_generators.items():
            self.data[name] = generator()
        return self.data

# Example usage
if __name__ == "__main__":
    generator = DataGenerator(1000, seed=42)
    generator.add_numerical("age", min_value=18, max_value=80, distribution="normal")
    generator.add_categorical("gender", categories=["Male", "Female", "Other"], weights=[0.48, 0.48, 0.04])
    generator.add_datetime("registration_date", start_date="2020-01-01", end_date="2023-06-30")
    generator.add_text("comment", min_words=5, max_words=20)
    generator.add_email("email")
    generator.add_phone_number("phone")
    generator.add_dependent("salary", depends_on="age", func=lambda age: age * 1000 + np.random.normal(0, 5000, len(age)))
    generator.add_custom("id", lambda: [f"USER_{i:04d}" for i in range(1000)])

    df = generator.generate()
    print(df.head())
    print(df.dtypes)