# project_management/project.py
import base64
import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from utils.file_utils import encode_image, save_json, load_json, DateTimeEncoder
from utils.logger import PlatoLogger
from config import PROJECT_DIR

# Create a logger instance
logger = PlatoLogger().logger

class Project:
    """
    A class to manage data analysis and research projects.

    This class provides functionality to define project metadata, add notes,
    incorporate charts and dataframes, and generate markdown reports.

    Attributes:
        name (str): The name of the project.
        goal (str): The main goal or objective of the project.
        created_at (datetime): The creation date and time of the project.
        updated_at (datetime): The last update date and time of the project.
        notes (List[Dict]): A list of note entries, each containing a timestamp and content.
        artifacts (Dict): A dictionary to store various project artifacts (charts, dataframes, etc.).

    Methods:
        add_note: Add a note to the project.
        add_chart: Add a chart (as an image file) to the project.
        add_dataframe: Add a pandas DataFrame to the project.
        add_artifact: Add a custom artifact to the project.
        generate_markdown: Generate a markdown report of the project.
        save: Save the project to a JSON file.
        load: Load a project from a JSON file.
    """

    def __init__(self, name: str, goal: str, project_path: str):
        self.data_dir = os.path.join(PROJECT_DIR, "data")
        self.name = name
        self.goal = goal
        self.project_path = project_path
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.notes = []
        self.artifacts = {}


        logger.info(f"Project '{self.name}' initialized with goal: '{self.goal}'")  # Log project initialization

    def add_note(self, note: str):
        """Add a note to the project."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.notes.append({"timestamp": timestamp, "content": note})
        logger.info(f"Note added: {note}")  # Log the addition of the note


    def add_chart(self, chart_name: str, chart_path: str):
        """Add a chart to the project."""
        with open(chart_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        self.artifacts[chart_name] = encoded_string
        logger.info(f"Chart added: {chart_name}")  # Log the addition of the chart

    def save(self):
        """Save the project to a JSON file."""
        project_file = os.path.join(self.project_path, f"{self.name.replace(' ', '_')}.json")
        save_json(self.__dict__, project_file)

    def generate_markdown(self):
        """Generate a markdown report of the project."""
        report = []
        report.append(f"# {self.name}\n")
        report.append(f"## Goal\n{self.goal}\n")
        report.append("## Notes\n")
        for note in self.notes:
            report.append(f"- {note['timestamp']}: {note['content']}\n")
        report.append("## Artifacts\n")
        for name, artifact in self.artifacts.items():
            if isinstance(artifact, str) and artifact.startswith("data:image"):
                report.append(f"![{name}]({artifact})\n")
            else:
                report.append(f"- {name}: {artifact}\n")
        return "".join(report)

    @classmethod
    def load(cls, project_path: str):
        """Load a project from a JSON file."""
        project_data = load_json(project_path)
        project = cls.__new__(cls)
        project.__dict__.update(project_data)
        project.project_path = project_path
        return project

    def add_dataframe(self, df: pd.DataFrame, name: str):
        """Add a pandas DataFrame to the project."""
        df_path = os.path.join(self.data_dir, f"{name}.csv")
        df.to_csv(df_path, index=False)
        self.artifacts[name] = df_path
        logger.info(f"DataFrame added: {name}")

    def add_artifact(self, name: str, artifact_type: str, artifact):
        """Add a custom artifact to the project."""
        if artifact_type == "dataframe":
            self.add_dataframe(artifact, name)
        else:
            self.artifacts[name] = artifact
            logger.info(f"Artifact added: {name}")

