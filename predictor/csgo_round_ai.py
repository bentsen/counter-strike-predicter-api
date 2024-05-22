import os

import pandas as pd
from pydantic import BaseModel
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from predictor.config import RESOURCES_DIR, MODEL_DIR


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


def train_and_save_model():
    df = pd.read_csv(os.path.join(RESOURCES_DIR, 'csgo_round_snapshots.csv'))

    le = LabelEncoder()

    df['bomb_planted'] = df['bomb_planted'].astype(int)
    df = pd.get_dummies(df, columns=['map'], dtype=int)
    df["round_winner"] = le.fit_transform(df["round_winner"])

    X, y = df.drop(['round_winner'], axis=1), df['round_winner']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf_model = RandomForestClassifier(n_jobs=4)
    rf_model.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_model.pkl'))


def load_model():
    # Load the model from the file
    model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    return model


def process_data(game_data: GameData):
    data = {
        "time_left": float(game_data.matchData.roundTime),
        "ct_score": int(game_data.matchData.ctScore),
        "t_score": int(game_data.matchData.tScore),
        "bomb_planted": game_data.matchData.bombPlanted and 1 or 0,
        "ct_health": int(game_data.matchData.ctHealth),
        "t_health": int(game_data.matchData.tHealth),
        "ct_armor": int(game_data.matchData.ctArmor),
        "t_armor": int(game_data.matchData.tArmor),
        "ct_money": int(4000),
        "t_money": int(1000),
        "ct_helmets": 4,
        "t_helmets": 3,
        "ct_defuse_kits": 0,
        "ct_players_alive": len(game_data.ct),
        "t_players_alive": len(game_data.t),
        "ct_weapon_ak47": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'AK-47'),
        "t_weapon_ak47": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'AK-47'),
        "ct_weapon_aug": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Aug'),
        "t_weapon_aug": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Aug'),
        "ct_weapon_awp": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Awp'),
        "t_weapon_awp": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Awp'),
        "ct_weapon_bizon": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Bizon'),
        "t_weapon_bizon": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Bizon'),
        "ct_weapon_cz75auto": sum(1 for player in game_data.ct if player.loadout.secondaryWeapon.name == 'Cz 75'),
        "t_weapon_cz75auto": sum(1 for player in game_data.t if player.loadout.secondaryWeapon.name == 'Cz 75'),
        "ct_weapon_elite": sum(1 for player in game_data.ct if player.loadout.secondaryWeapon.name == 'Dual Berettas'),
        "t_weapon_elite": sum(1 for player in game_data.t if player.loadout.secondaryWeapon.name == 'Dual Berettas'),
        "ct_weapon_famas": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Famas'),
        "t_weapon_famas": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Famas'),
        "ct_weapon_g3sg1": 0,
        "t_weapon_g3sg1": 0,
        "ct_weapon_galilar": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Galil'),
        "t_weapon_galilar": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Galil'),
        "ct_weapon_glock": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Glock'),
        "t_weapon_glock": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Glock'),
        "ct_weapon_m249": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'M249'),
        "t_weapon_m249": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'M249'),
        "ct_weapon_m4a1s": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'M4a1'),
        "t_weapon_m4a1s": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'MM4a1'),
        "ct_weapon_m4a4": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'M4a4'),
        "t_weapon_m4a4": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'M4a4'),
        "ct_weapon_mac10": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Mac 10'),
        "t_weapon_mac10": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Mac 10'),
        "ct_weapon_mag7": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Mag 7'),
        "t_weapon_mag7": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Mag 7'),
        "ct_weapon_mp5sd": 0,
        "t_weapon_mp5sd": 0,
        "ct_weapon_mp7": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Mp7'),
        "t_weapon_mp7": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Mp7'),
        "ct_weapon_mp9": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Mp9'),
        "t_weapon_mp9": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Mp9'),
        "ct_weapon_negev": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Negev'),
        "t_weapon_negev": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Negev'),
        "ct_weapon_nova": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Nova'),
        "t_weapon_nova": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Nova'),
        "ct_weapon_p90": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'P90'),
        "t_weapon_p90": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'P90'),
        "ct_weapon_r8revolver": 0,
        "t_weapon_r8revolver": 0,
        "ct_weapon_sawedoff": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Sawed Off'),
        "t_weapon_sawedoff": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Sawed Off'),
        "ct_weapon_scar20": 0,
        "t_weapon_scar20": 0,
        "ct_weapon_sg553": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Sg 553'),
        "t_weapon_sg553": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Sg 553'),
        "ct_weapon_ssg08": 0,
        "t_weapon_ssg08": 0,
        "ct_weapon_ump45": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Ump 45'),
        "t_weapon_ump45": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Ump 45'),
        "ct_weapon_xm1014": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Xm1014'),
        "t_weapon_xm1014": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Xm1014'),
        "ct_weapon_deagle": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Deagle'),
        "t_weapon_deagle": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Deagle'),
        "ct_weapon_fiveseven": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Five Seven'),
        "t_weapon_fiveseven": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Five Seven'),
        "ct_weapon_usps": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Usp S'),
        "t_weapon_usps": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Usp S'),
        "ct_weapon_p250": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'P250'),
        "t_weapon_p250": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'P250'),
        "ct_weapon_p2000": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'P2000'),
        "t_weapon_p2000": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'P2000'),
        "ct_weapon_tec9": sum(1 for player in game_data.ct if player.loadout.mainWeapon.name == 'Tec 9'),
        "t_weapon_tec9": sum(1 for player in game_data.t if player.loadout.mainWeapon.name == 'Tec 9'),
        "ct_grenade_hegrenade": sum(1 for Nade in game_data.ct if (nade.name == 'He Grenade' for nade in
                                                                   [Nade.loadout.utility.nade1,
                                                                    Nade.loadout.utility.nade2,
                                                                    Nade.loadout.utility.nade3,
                                                                    Nade.loadout.utility.nade4])),
        "t_grenade_hegrenade": sum(1 for Nade in game_data.t if (nade.name == 'He Grenade' for nade in
                                                                 [Nade.loadout.utility.nade1,
                                                                  Nade.loadout.utility.nade2,
                                                                  Nade.loadout.utility.nade3,
                                                                  Nade.loadout.utility.nade4])),
        "ct_grenade_flashbang": sum(1 for Nade in game_data.ct if (nade.name == 'Flashbang' for nade in
                                                                   [Nade.loadout.utility.nade1,
                                                                    Nade.loadout.utility.nade2,
                                                                    Nade.loadout.utility.nade3,
                                                                    Nade.loadout.utility.nade4])),
        "t_grenade_flashbang": sum(1 for Nade in game_data.t if (nade.name == 'Flashbang' for nade in
                                                                 [Nade.loadout.utility.nade1,
                                                                  Nade.loadout.utility.nade2,
                                                                  Nade.loadout.utility.nade3,
                                                                  Nade.loadout.utility.nade4])),
        "ct_grenade_smokegrenade": sum(1 for Nade in game_data.ct if (nade.name == 'Smoke' for nade in
                                                                      [Nade.loadout.utility.nade1,
                                                                       Nade.loadout.utility.nade2,
                                                                       Nade.loadout.utility.nade3,
                                                                       Nade.loadout.utility.nade4])),
        "t_grenade_smokegrenade": sum(1 for Nade in game_data.t if (nade.name == 'Smoke' for nade in
                                                                    [Nade.loadout.utility.nade1,
                                                                     Nade.loadout.utility.nade2,
                                                                     Nade.loadout.utility.nade3,
                                                                     Nade.loadout.utility.nade4])),
        "ct_grenade_incendiarygrenade": sum(1 for Nade in game_data.ct if (nade.name == 'Incendiary' for nade in
                                                                           [Nade.loadout.utility.nade1,
                                                                            Nade.loadout.utility.nade2,
                                                                            Nade.loadout.utility.nade3,
                                                                            Nade.loadout.utility.nade4])),
        "t_grenade_incendiarygrenade": sum(1 for Nade in game_data.t if (nade.name == 'Incendiary' for nade in
                                                                         [Nade.loadout.utility.nade1,
                                                                          Nade.loadout.utility.nade2,
                                                                          Nade.loadout.utility.nade3,
                                                                          Nade.loadout.utility.nade4])),
        "ct_grenade_molotovgrenade": sum(1 for Nade in game_data.ct if (nade.name == 'Molotov' for nade in
                                                                        [Nade.loadout.utility.nade1,
                                                                         Nade.loadout.utility.nade2,
                                                                         Nade.loadout.utility.nade3,
                                                                         Nade.loadout.utility.nade4])),
        "t_grenade_molotovgrenade": sum(1 for Nade in game_data.t if (nade.name == 'Molotov' for nade in
                                                                      [Nade.loadout.utility.nade1,
                                                                       Nade.loadout.utility.nade2,
                                                                       Nade.loadout.utility.nade3,
                                                                       Nade.loadout.utility.nade4])),
        "ct_grenade_decoygrenade": sum(1 for Nade in game_data.ct if (nade.name == 'Decoy' for nade in
                                                                      [Nade.loadout.utility.nade1,
                                                                       Nade.loadout.utility.nade2,
                                                                       Nade.loadout.utility.nade3,
                                                                       Nade.loadout.utility.nade4])),
        "t_grenade_decoygrenade": sum(1 for Nade in game_data.t if (nade.name == 'Decoy' for nade in
                                                                    [Nade.loadout.utility.nade1,
                                                                     Nade.loadout.utility.nade2,
                                                                     Nade.loadout.utility.nade3,
                                                                     Nade.loadout.utility.nade4])),
        "map_de_cache": game_data.selectedMap == "Cache" and 1 or 0,
        "map_de_dust2": game_data.selectedMap == "Dust 2" and 1 or 0,
        "map_de_inferno": game_data.selectedMap == "Inferno" and 1 or 0,
        "map_de_mirage": game_data.selectedMap == "Mirage" and 1 or 0,
        "map_de_nuke": game_data.selectedMap == "Nuke" and 1 or 0,
        "map_de_overpass": game_data.selectedMap == "Overpass" and 1 or 0,
        "map_de_train": game_data.selectedMap == "Train" and 1 or 0,
        "map_de_vertigo": game_data.selectedMap == "Vertigo" and 1 or 0,
    }
    return data


def predict_round(processed_data):
    model = load_model()
    input_df = pd.DataFrame(processed_data, index=[0])

    [rd_probabilities_ct, rd_probabilities_t] = model.predict_proba(input_df)[0]
    return {
        "ct": (rd_probabilities_ct * 100),
        "t": rd_probabilities_t * 100
    }