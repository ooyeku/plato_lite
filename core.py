# core.py

import os
from typing import Optional, List, Dict, Any

import numpy as np

from config import PROJECT_DIR, DATA_DIR, DB_PATH
from utils.logger import logger
from data_management.data_manager import DataManager
from insights.insights import Insights
from project_management.project import Project
from utils.data_generator import DataGenerator


class Plato:
    """
    Core class for the Plato system, integrating data management, insights, and project management.
    """

    def __init__(self):
        self.data_manager = DataManager(DB_PATH)
        self.current_project: Optional[Project] = None


    def create_project(self, name: str, goal: str) -> Project:
        """Create a new project."""
        project_path = os.path.join(PROJECT_DIR, f"project_{name.replace(' ', '_')}")
        os.makedirs(project_path, exist_ok=True)
        project = Project(name, goal, project_path)
        self.current_project = project
        logger.info(f"Created new project: {name}")
        return project

    def load_project(self, name: str) -> Project:
        """Load an existing project."""
        project_path = os.path.join(PROJECT_DIR, f"{name.replace(' ', '_')}.json")
        project = Project.load(project_path)
        self.current_project = project
        logger.info(f"Loaded project: {name}")
        return project

    def load_data(self, file_path: str, table_name: Optional[str] = None) -> Any:
        """Load data from a file and optionally save to database."""
        data = self.data_manager.load_data(file_path, table_name)
        if self.current_project:
            self.current_project.add_note(f"Loaded data from {file_path}")
        logger.info(f"Loaded data from {file_path}")
        return data

    def analyze(self, data: Any, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Perform analysis on data."""
        insights = Insights(data)
        result = getattr(insights, analysis_type)(**kwargs)

        if self.current_project:
            self.current_project.add_note(f"Performed {analysis_type} analysis")
            if isinstance(result, dict):
                for key, value in result.items():
                    self.current_project.add_artifact(f"{analysis_type}_{key}", "analysis_result", value)

        logger.info(f"Performed {analysis_type} analysis")
        return result

    def visualize(self, data: Any, viz_type: str, **kwargs) -> None:
        """Create visualization from data."""
        insights = Insights(data)
        method_name = f"plot_{viz_type}"
        if hasattr(insights, method_name):
            fig = getattr(insights, method_name)(**kwargs)
        else:
            raise AttributeError(f"'Insights' object has no attribute '{method_name}'")

        if self.current_project:
            viz_path = os.path.join(DATA_DIR, f"{viz_type}_plot.png")
            fig.savefig(viz_path)
            self.current_project.add_chart(f"{viz_type}_plot", viz_path)
            self.current_project.add_note(f"Created {viz_type} visualization")

        logger.info(f"Created {viz_type} visualization")

    def save_project(self) -> None:
        """Save the current project."""
        if self.current_project:
            self.current_project.save()
            logger.info(f"Saved project: {self.current_project.name}")
        else:
            logger.warning("No active project to save")

    def generate_report(self) -> Optional[str]:
        """Generate a report for the current project."""
        if self.current_project:
            report = self.current_project.generate_markdown()
            logger.info("Generated project report")
            return report
        else:
            logger.warning("No active project to generate report")
            return None


# Example usage
if __name__ == "__main__":
    plato = Plato()

    # Create a new project
    project = plato.create_project("Sample Analysis", "Analyze sample data")

    # generate sample data
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

    project_data_path = os.path.join(project.data_dir, "sample_table.csv")

    # save data
    plato.data_manager.save_data(df, project_data_path)

    # Load data
    data = plato.load_data(project_data_path, "sample_table")

    # Perform analysis
    analysis_result = plato.analyze(data, "descriptive_statistics")

    # Create visualization
    plato.visualize(data, "scatter", x="age", y="salary")

    # Save project
    plato.save_project()

    # Generate report
    report = plato.generate_report()
    print(report)



