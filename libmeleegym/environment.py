import abc
import atexit
import gym
import logging
import melee

from dataclasses import dataclass
from gym import spaces
from typing import Dict, Mapping

from libmeleegym import embed
from libmeleegym.melee_gym_config import MeleeGymConfig

"""
  Gather combos and other positive affects, collect the frames for that to act in reward calculation.
  If GAIL can be used on imitation data.
"""

class Player(abc.ABC):

  @abc.abstractmethod
  def controller_type(self) -> melee.ControllerType:
    pass

  @abc.abstractmethod
  def menuing_kwargs(self) -> Dict:
    pass


class Human(Player):

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.GCN_ADAPTER

  def menuing_kwargs(self) -> Dict:
      return {}

@dataclass
class CPU(Player):
  character: melee.Character = melee.Character.FOX
  level: int = 9

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character, cpu_level=self.level)

@dataclass
class AI(Player):
  character: melee.Character = melee.Character.FOX

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character)

def send_controller(controller: melee.Controller, controller_state: dict):
  for b in embed.LEGAL_BUTTONS:
    if controller_state['button'][b.value]:
      controller.press_button(b)
    else:
      controller.release_button(b)
  main_stick = controller_state["main_stick"]
  controller.tilt_analog(melee.Button.BUTTON_MAIN, *main_stick)
  c_stick = controller_state["c_stick"]
  controller.tilt_analog(melee.Button.BUTTON_C, *c_stick)
  controller.press_shoulder(melee.Button.BUTTON_L, controller_state["l_shoulder"])
  controller.press_shoulder(melee.Button.BUTTON_R, controller_state["r_shoulder"])

def _is_menu_state(gamestate: melee.GameState) -> bool:
  return gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]

class MeleeGymEnv(gym.Env):

  def __init__(self,
               config : MeleeGymConfig,
               embedder : embed.Embedding,
               players : Mapping[int, Player]):
    super(MeleeGymEnv, self).__init__()
    self.__embedder = embedder
    self.__players = players
    self.__config = config.__dict__
    self.__stage = config.stage

    if config.headless:
      self.__config.update({
        'render': False,
        'disable_audio': True,
        'use_exi_inputs': True,
        'enable_ffw': True,
      })

    self.__dolphin_console = melee.Console(
      **{k: v for k, v in self.__config.items() if hasattr(melee.Console, k)}
    )
    atexit.register(self.__dolphin_console.stop)
    self._menuing_controllers = []
    self._autostart = True
    self.controllers = {}

    for port, player in self.__players.items():
      controller = melee.Controller(
        self.__dolphin_console, port, player.controller_type())
      self.controllers[port] = controller
      assert not isinstance(player, Human), 'Cannot have player as Human, not implemented yet.'
      self._menuing_controllers.append((controller, player))

    self.actions_space = None
    self.observation_space = None


  def _next_gamestate(self) -> melee.GameState:
    gamestate = self.__dolphin_console.step()
    assert gamestate is not None
    return gamestate


  def reset(self):
    self.__dolphin_console.stop()
    self.__dolphin_console.run(
      iso_path=self.__config['melee_iso'],
      environment_vars=self.__config['env_vars'],
    )

    logging.info('Connecting to console...')
    if not self.__dolphin_console.connect():
      raise RuntimeError("Failed to connect to the console.")

    for controller in self.controllers.values():
      if not controller.connect():
        raise RuntimeError("Failed to connect the controller.")

    gamestate = self._next_gamestate()

    # The console object keeps track of how long your bot is taking to process frames
    #   And can warn you if it's taking too long
    # if self.console.processingtime * 1000 > 12:
    #     print("WARNING: Last frame took " + str(self.console.processingtime*1000) + "ms to process.")

    menu_frames = 0
    while _is_menu_state(gamestate):
      for i, (controller, player) in enumerate(self._menuing_controllers):

        melee.MenuHelper.menu_helper_simple(
            gamestate, controller,
            stage_selected=self.__stage,
            connect_code=None,
            autostart=self._autostart and i == 0 and menu_frames > 180,
            swag=False,
            costume=i,
            **player.menuing_kwargs())

        gamestate = self._next_gamestate()
        menu_frames += 1

    return self.__embedder.from_state(gamestate)


  def step(self, action: Mapping[int, dict]):
    state = None
    reward = None
    info = {}
    for port, controller in self.controllers.items():
      send_controller(controller, action[port])

    gamestate = self._next_gamestate()
    done = _is_menu_state(gamestate)
    if not done:
      state = self.__embedder.from_state(gamestate)
      # Extract controller specifics?

      # Extract state
    return state, reward, info, done


  def render(self, mode="human"):
    pass


  def close(self):
    self.__dolphin_console.stop()