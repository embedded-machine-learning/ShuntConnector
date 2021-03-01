import configparser
from pathlib import Path
import sys

from shunt_connector import ShuntConnector

config_name = "standard.cfg"

if len(sys.argv) > 1:
    config_name = sys.argv[1]

config_path = Path(sys.path[0], "config", config_name)
config = configparser.ConfigParser()
config.read(config_path)

connector = ShuntConnector(config)
connector.execute()
