import os

import numpy as np
import pandas as pd

import etdtransform


def get_project_weather_station_data():
    """
    Load and process project weather station data.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing project weather station data with uppercase station names.
    """
    # Load mapping of ProjectIdBSV to weather stations and ensure names are uppercase
    weather_station_file = etdtransform.options.weather_stations_summary_file
    project_weather_station_df = pd.read_excel(
        weather_station_file,
        sheet_name="ProjectWeatherStation",
    )
    project_weather_station_df["Weerstation"] = project_weather_station_df[
        "Weerstation"
    ].str.upper()
    project_weather_station_df["STN"] = project_weather_station_df["Nummer"]

    return project_weather_station_df


def get_weather_data():
    """
    Load and process weather data from CSV files.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing combined weather data from all CSV files.
    """
    # Load all temperature CSV files from the folder
    weather_data_folder = etdtransform.options.weather_data_folder_path

    weather_data_df = load_knmi_weather_data(weather_data_folder)

    return weather_data_df


def load_knmi_weather_data(folder_path):
    """
    Load and process KNMI weather data from text files in a specified folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing KNMI weather data files.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing combined and processed weather data from all files.
    """
    combined_df = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            # Count the number of commented lines
            with open(file_path, "r") as file:
                commented_lines = 0
                for line in file:
                    if line.startswith("#"):
                        commented_lines += 1
            # Keep last commented line, and load the data with that line as the header
            df = pd.read_csv(file_path, skiprows=commented_lines - 1, header=0)
            df.columns = df.columns.str.strip()
            df.rename(columns={"# STN": "STN"}, inplace=True)
            df["Temperatuur"] = df["T"] / 10  # noqa E501 Convert temperature to degrees Celsius
            df["Windsnelheid"] = df["FH"] / 10  # Convert wind speed to m/s
            df["Vochtigheid"] = df["U"]  # Humidity is already in percentage
            humidity_coefficient = 0.33  # Replace with local value if available
            wind_speed_adjustment = 4.00  # Replace with local value if available
            vapor_pressure_constant = 17.27  # Replace with local value if available
            wind_speed_coefficient = 0.7  # Replace with local value if available
            df["Dampdruk"] = (
                df["Vochtigheid"]
                * 6.105
                * np.exp(
                    (vapor_pressure_constant * df["Temperatuur"])
                    / (df["Temperatuur"] + 237.7),
                )
                / 100
            )
            df["Gevoelstemperatuur"] = (
                df["Temperatuur"]
                + humidity_coefficient * df["Dampdruk"]
                - wind_speed_coefficient * df["Windsnelheid"]
                - wind_speed_adjustment
            )
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df


weather_columns = [
    "STN",
    "YYYYMMDD",
    "HH",
    "Temperatuur",
    "Gevoelstemperatuur",
    "Vochtigheid",
    "Windsnelheid",
    "TemperatuurRA",
    "GevoelstemperatuurRA",
    "Koudste2WkTemperatuur",
    "Koudste2WkGevoelstemperatuur",
]
