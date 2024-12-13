from collections import namedtuple

from etdmap._config import Options

Option = namedtuple("Option", "key default_value doc validator callback")

# Define allowed Options
mapped_folder_path = Option(
    key="mapping_folder_path",
    # By default it is stored locally in the data folder.
    default_value=r"./data/mapped",
    doc=(
        "The folder containing the mapped (input) data files"
    ),
    validator=None,
    callback=None,
)

aggregate_folder_path = Option(
    key="aggregate_folder_path",
    default_value=r"./data/aggregate",
    doc=(
        "The folder containing aggregated (output) files"
    ),
    validator=None,
    callback=None,
)

weather_data_folder_path = Option(
    key="weather_data_folder_path",
    default_value=None,
    doc=(
        "The folder containing the knmi files for the stations."
        "The files are downloaded from https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
    ),
    validator=None,
    callback=None,
)

weather_stations_summary_file = Option(
    key="weather_stations_summary_file",
    default_value=None,
    doc=(
        "# path to file that contains the list of the stations and their codes"
    ),
    validator=None,
    callback=None,
)

# Set the option with default values
options = Options(
    {
        "mapped_folder_path": mapped_folder_path,
        "aggregate_folder_path": aggregate_folder_path,
        "weather_data_folder_path": weather_data_folder_path,
        "weather_stations_summary_file": weather_stations_summary_file,
    }
)
