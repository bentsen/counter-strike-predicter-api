import os

import joblib
import pandas as pd
import logging

from src.predictors.graph_functions import generate_graphs
from src.predictors.train_model import train_and_save_model
from src.types.round_type import GameData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model():
    try:
        model = joblib.load(os.path.join('src', 'predictors', 'rf_model.pkl'))
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def process_data(game_data: GameData):
    data = {
        "time_left": float(game_data.matchData.roundTime),
        "ct_score": int(game_data.matchData.ctScore),
        "t_score": int(game_data.matchData.tScore),
        "bomb_planted": int(game_data.matchData.bombPlanted),
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
        "ct_weapon_ak47": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'AK-47'),
        "t_weapon_ak47": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'AK-47'),
        "ct_weapon_aug": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Aug'),
        "t_weapon_aug": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Aug'),
        "ct_weapon_awp": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Awp'),
        "t_weapon_awp": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Awp'),
        "ct_weapon_bizon": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Bizon'),
        "t_weapon_bizon": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Bizon'),
        "ct_weapon_cz75auto": sum(1 for player in game_data.ct if
                                  player.loadout.secondaryWeapon and player.loadout.secondaryWeapon.name == 'Cz 75'),
        "t_weapon_cz75auto": sum(1 for player in game_data.t if
                                 player.loadout.secondaryWeapon and player.loadout.secondaryWeapon.name == 'Cz 75'),
        "ct_weapon_elite": sum(1 for player in game_data.ct if
                               player.loadout.secondaryWeapon and player.loadout.secondaryWeapon.name == 'Dual Berettas'),
        "t_weapon_elite": sum(1 for player in game_data.t if
                              player.loadout.secondaryWeapon and player.loadout.secondaryWeapon.name == 'Dual Berettas'),
        "ct_weapon_famas": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Famas'),
        "t_weapon_famas": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Famas'),
        "ct_weapon_g3sg1": 0,
        "t_weapon_g3sg1": 0,
        "ct_weapon_galilar": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Galil'),
        "t_weapon_galilar": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Galil'),
        "ct_weapon_glock": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Glock'),
        "t_weapon_glock": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Glock'),
        "ct_weapon_m249": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'M249'),
        "t_weapon_m249": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'M249'),
        "ct_weapon_m4a1s": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'M4a1'),
        "t_weapon_m4a1s": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'MM4a1'),
        "ct_weapon_m4a4": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'M4a4'),
        "t_weapon_m4a4": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'M4a4'),
        "ct_weapon_mac10": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mac 10'),
        "t_weapon_mac10": sum(
            1 for player in game_data.t if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mac 10'),
        "ct_weapon_mag7": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mag 7'),
        "t_weapon_mag7": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mag 7'),
        "ct_weapon_mp5sd": 0,
        "t_weapon_mp5sd": 0,
        "ct_weapon_mp7": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mp7'),
        "t_weapon_mp7": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mp7'),
        "ct_weapon_mp9": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mp9'),
        "t_weapon_mp9": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Mp9'),
        "ct_weapon_negev": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Negev'),
        "t_weapon_negev": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Negev'),
        "ct_weapon_nova": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Nova'),
        "t_weapon_nova": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Nova'),
        "ct_weapon_p90": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'P90'),
        "t_weapon_p90": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'P90'),
        "ct_weapon_r8revolver": 0,
        "t_weapon_r8revolver": 0,
        "ct_weapon_sawedoff": sum(1 for player in game_data.ct if
                                  player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Sawed Off'),
        "t_weapon_sawedoff": sum(
            1 for player in game_data.t if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Sawed Off'),
        "ct_weapon_scar20": 0,
        "t_weapon_scar20": 0,
        "ct_weapon_sg553": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Sg 553'),
        "t_weapon_sg553": sum(
            1 for player in game_data.t if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Sg 553'),
        "ct_weapon_ssg08": 0,
        "t_weapon_ssg08": 0,
        "ct_weapon_ump45": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Ump 45'),
        "t_weapon_ump45": sum(
            1 for player in game_data.t if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Ump 45'),
        "ct_weapon_xm1014": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Xm1014'),
        "t_weapon_xm1014": sum(
            1 for player in game_data.t if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Xm1014'),
        "ct_weapon_deagle": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Deagle'),
        "t_weapon_deagle": sum(
            1 for player in game_data.t if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Deagle'),
        "ct_weapon_fiveseven": sum(1 for player in game_data.ct if
                                   player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Five Seven'),
        "t_weapon_fiveseven": sum(1 for player in game_data.t if
                                  player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Five Seven'),
        "ct_weapon_usps": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Usp S'),
        "t_weapon_usps": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Usp S'),
        "ct_weapon_p250": sum(
            1 for player in game_data.ct if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'P250'),
        "t_weapon_p250": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'P250'),
        "ct_weapon_p2000": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'P2000'),
        "t_weapon_p2000": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'P2000'),
        "ct_weapon_tec9": sum(
            1 for player in game_data.ct if
            player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Tec 9'),
        "t_weapon_tec9": sum(
            1 for player in game_data.t if player.loadout.mainWeapon and player.loadout.mainWeapon.name == 'Tec 9'),
        "ct_grenade_hegrenade": sum(1 for Nade in game_data.ct if any(nade and nade.name == 'He Grenade' for nade in
                                                                      [Nade.loadout.utility.nade1,
                                                                       Nade.loadout.utility.nade2,
                                                                       Nade.loadout.utility.nade3,
                                                                       Nade.loadout.utility.nade4])),
        "t_grenade_hegrenade": sum(1 for Nade in game_data.t if any(nade and nade.name == 'He Grenade' for nade in
                                                                    [Nade.loadout.utility.nade1,
                                                                     Nade.loadout.utility.nade2,
                                                                     Nade.loadout.utility.nade3,
                                                                     Nade.loadout.utility.nade4])),
        "ct_grenade_flashbang": sum(1 for Nade in game_data.ct if any(nade and nade.name == 'Flashbang' for nade in
                                                                      [Nade.loadout.utility.nade1,
                                                                       Nade.loadout.utility.nade2,
                                                                       Nade.loadout.utility.nade3,
                                                                       Nade.loadout.utility.nade4])),
        "t_grenade_flashbang": sum(1 for Nade in game_data.t if any(nade and nade.name == 'Flashbang' for nade in
                                                                    [Nade.loadout.utility.nade1,
                                                                     Nade.loadout.utility.nade2,
                                                                     Nade.loadout.utility.nade3,
                                                                     Nade.loadout.utility.nade4])),
        "ct_grenade_smokegrenade": sum(1 for Nade in game_data.ct if any(nade and nade.name == 'Smoke' for nade in
                                                                         [Nade.loadout.utility.nade1,
                                                                          Nade.loadout.utility.nade2,
                                                                          Nade.loadout.utility.nade3,
                                                                          Nade.loadout.utility.nade4])),
        "t_grenade_smokegrenade": sum(1 for Nade in game_data.t if any(nade and nade.name == 'Smoke' for nade in
                                                                       [Nade.loadout.utility.nade1,
                                                                        Nade.loadout.utility.nade2,
                                                                        Nade.loadout.utility.nade3,
                                                                        Nade.loadout.utility.nade4])),
        "ct_grenade_incendiarygrenade": sum(
            1 for Nade in game_data.ct if any(nade and nade.name == 'Incendiary' for nade in
                                              [Nade.loadout.utility.nade1,
                                               Nade.loadout.utility.nade2,
                                               Nade.loadout.utility.nade3,
                                               Nade.loadout.utility.nade4])),
        "t_grenade_incendiarygrenade": sum(
            1 for Nade in game_data.t if any(nade and nade.name == 'Incendiary' for nade in
                                             [Nade.loadout.utility.nade1,
                                              Nade.loadout.utility.nade2,
                                              Nade.loadout.utility.nade3,
                                              Nade.loadout.utility.nade4])),
        "ct_grenade_molotovgrenade": sum(
            1 for Nade in game_data.ct if any(nade and nade.name == 'Molotov' for nade in
                                              [Nade.loadout.utility.nade1,
                                               Nade.loadout.utility.nade2,
                                               Nade.loadout.utility.nade3,
                                               Nade.loadout.utility.nade4])),
        "t_grenade_molotovgrenade": sum(1 for Nade in game_data.t if any(nade and nade.name == 'Molotov' for nade in
                                                                         [Nade.loadout.utility.nade1,
                                                                          Nade.loadout.utility.nade2,
                                                                          Nade.loadout.utility.nade3,
                                                                          Nade.loadout.utility.nade4])),
        "ct_grenade_decoygrenade": sum(1 for Nade in game_data.ct if any(nade and nade.name == 'Decoy' for nade in
                                                                         [Nade.loadout.utility.nade1,
                                                                          Nade.loadout.utility.nade2,
                                                                          Nade.loadout.utility.nade3,
                                                                          Nade.loadout.utility.nade4])),
        "t_grenade_decoygrenade": sum(1 for Nade in game_data.t if any(nade and nade.name == 'Decoy' for nade in
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


class CSGORoundPredictor:
    def __init__(self):
        model_path = os.path.join('src', 'predictors', 'rf_model.pkl')
        if not os.path.exists(model_path):
            logger.info("Model file not found. Training a new model...")
            train_and_save_model()

        self.model = load_model()
        logger.info("Model loaded successfully.")

        graph_path = os.path.join('src', 'graphs', 'round_predictor')
        if not os.path.exists(graph_path):
            logger.info("Graph file not found. Creating a new graph...")
            os.makedirs(graph_path)
            generate_graphs()

    def predict_round(self, game_data: GameData):
        try:
            processed_data = process_data(game_data)
            input_df = pd.DataFrame(processed_data, index=[0])
            rd_probabilities_ct, rd_probabilities_t = self.model.predict_proba(input_df)[0]
            return {
                "ct": rd_probabilities_ct * 100,
                "t": rd_probabilities_t * 100
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
