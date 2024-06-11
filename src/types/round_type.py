from pydantic import BaseModel
from typing import List, Dict


class Weapon(BaseModel):
    id: int
    name: str
    img: str


class Utility(BaseModel):
    nade1: Weapon | None
    nade2: Weapon | None
    nade3: Weapon | None
    nade4: Weapon | None


class Loadout(BaseModel):
    mainWeapon: Weapon | None
    secondaryWeapon: Weapon | None
    utility: Utility


class Team(BaseModel):
    loadout: Loadout


class MatchData(BaseModel):
    tScore: str
    ctScore: str
    roundTime: str
    tHealth: str
    tArmor: str
    bombPlanted: bool
    ctHealth: str
    ctArmor: str


class GameData(BaseModel):
    t: List[Team]
    ct: List[Team]
    selectedMap: str
    matchData: MatchData
