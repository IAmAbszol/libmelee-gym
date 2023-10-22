import json
import melee

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MeleeGymConfig:
  """A dataclass for holding configuration values for a Melee Gym environment using libmelee.

  Attributes:
    environment_name (str): The name of the Gym environment. Default is "Melee Gym Environment".
    dolphin_path (str): The path to the directory where your dolphin executable is located.
    melee_iso (str): The path to the Melee ISO for dolphin to read.
    stage (melee.Stage): The stage to be set for the game. Default is FINAL_DESTINATION.
    online_delay (int): How many frames of delay to apply in online matches. Default is 0.
    blocking_input (bool): Should dolphin block waiting for bot input. Default is True.
    polling_mode (bool): Polls input to console rather than blocking for it. Default is False.
    headless (bool): Run the emulator without rendering the game on screen. Default is True.
    save_replays (bool): Save Slippi replays. Default is False.
    env_vars (List[str]): List of environment variables to set for Dolphin. Default is None.
    overclock (float, optional): Overclock the Dolphin CPU. Default is None.
    num_episodes (int): The number of episodes for the agent to run. Default is 100.
    learning_rate (float): The learning rate for the agent. Default is 1e-5.

  Example:
    >>> config = MeleeGymConfig.from_json("config.json")
    >>> print(config.environment_name)
    'MyMeleeEnvironment'
  """

  environment_name: str = "Melee Gym Environment"
  dolphin_path: str = "/path/to/dolphin"
  melee_iso: str = "/path/to/melee.iso"
  stage: melee.Stage = melee.Stage.FINAL_DESTINATION
  online_delay: int = 0
  blocking_input: bool = True
  polling_mode: bool = False
  headless: bool = True
  save_replays: bool = False
  env_vars: Optional[List[str]] = None
  overclock: Optional[float] = None
  num_episodes: int = 100
  learning_rate: float = 1e-5

  @classmethod
  def from_json(cls, json_file_path: str) -> 'MeleeGymConfig':
    """Creates a MeleeGymConfig object from a JSON file.

    Args:
      json_file_path (str): The path to the JSON file containing the configuration.

    Returns:
      MeleeGymConfig: A MeleeGymConfig object initialized with values from the JSON file.

    Example:
      >>> config = MeleeGymConfig.from_json("config.json")
      >>> print(config.num_episodes)
      1000

    Raises:
      FileNotFoundError: If the specified JSON file does not exist.
      json.JSONDecodeError: If the JSON file is not properly formatted.
    """

    with open(json_file_path, 'r') as fd:
      config_dict = json.load(fd)
    return cls(**config_dict)
