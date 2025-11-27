import os
import re
import glob
import joblib
import hashlib

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from sklearn.neighbors import KDTree


class FeatureEngineer:
    """
    A class to handle feature engineering for NFL player tracking data.
    Supports caching intermediate results and applying features independently.
    """

    def __init__(self, data_path: str = "", cache_dir: str = "cache"):
        """
        Initialize the FeatureEngineer with data paths and cache directory.

        Args:
            data_path (str): Path to the directory containing input/output CSV files.
            cache_dir (str): Directory to store cached intermediate results.
        """
        self.data_path = data_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.X_LIMIT = 120.0
        self.Y_LIMIT = 53.3
        self.input_df: Optional[pd.DataFrame] = None
        self.output_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []

    def _cache_key(self, step_name: str, **kwargs) -> str:
        """
        Generate a unique cache key for a given step and its parameters.

        Args:
            step_name (str): Name of the feature engineering step.
            **kwargs: Parameters used in the step.

        Returns:
            str: A unique hash key for caching.
        """
        key_str = f"{step_name}_{kwargs}".encode("utf-8")
        return hashlib.md5(key_str).hexdigest()

    def _load_or_compute(
        self, step_name: str, func, force_recompute: bool = False, **kwargs
    ):
        """
        Load cached results or compute and cache them if they don't exist.

        Args:
            step_name (str): Name of the step (used for caching).
            func (callable): Function to compute the result.
            force_recompute (bool): If True, recompute even if cache exists.
            **kwargs: Arguments to pass to `func`.

        Returns:
            Result of `func(**kwargs)`.
        """
        cache_key = self._cache_key(step_name, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not force_recompute and cache_file.exists():
            return joblib.load(cache_file)
        else:
            result = func(**kwargs)
            joblib.dump(result, cache_file)
            return result

    def load_data(self, force_recompute: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and concatenate input and output CSV files.

        Args:
            force_recompute (bool): If True, reload data even if cached.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Concatenated input and output DataFrames.
        """
        def _load_data():
            files_input = sorted(glob.glob(os.path.join(self.data_path, "input_2023_w*.csv")))
            files_output = sorted(glob.glob(os.path.join(self.data_path, "output_2023_w*.csv")))

            print(f"Found {len(files_input)} input files: {files_input[:3]}...")
            print(f"Found {len(files_output)} output files: {files_output[:3]}...")

            if not files_input or not files_output:
                raise FileNotFoundError(f"No files found in {self.data_path}")

            input_dfs, output_dfs = [], []
            for f_in, f_out in zip(files_input, files_output):
                df_in = pd.read_csv(f_in)
                df_out = pd.read_csv(f_out)

                print(f"Reading {f_in}: shape {df_in.shape}")
                print(f"Reading {f_out}: shape {df_out.shape}")

                if df_in.empty or df_out.empty:
                    print(f"Warning: Empty file {f_in} or {f_out}")
                    continue

                week_match = re.search(r"w(\d+)", f_out)
                week = int(week_match.group(1)) if week_match else None
                df_in["week"] = week
                df_out["week"] = week
                input_dfs.append(df_in)
                output_dfs.append(df_out)

            if not input_dfs or not output_dfs:
                raise ValueError("No valid data to concatenate. Check files for emptiness.")

            return pd.concat(input_dfs, ignore_index=True), pd.concat(output_dfs, ignore_index=True)

        try:
            self.input_df, self.output_df = self._load_or_compute(
                "load_data", _load_data, force_recompute=force_recompute
            )
        except Exception as e:
            print(f"Error during data loading: {e}")
            raise
        return self.input_df, self.output_df


    def create_training_rows(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Create training rows by merging input and output data.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: Merged DataFrame with target columns.
        """
        def _create_training_rows(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
            agg = (
                input_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
                .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
                .tail(1)
                .reset_index(drop=True)
                .rename(columns={"frame_id": "last_frame_id"})
            )
            out = output_df.copy()
            out = out.rename(columns={"x": "target_x", "y": "target_y"})
            out["id"] = (
                out["game_id"].astype(str) + "_" +
                out["play_id"].astype(str) + "_" +
                out["nfl_id"].astype(str) + "_" +
                out["frame_id"].astype(str)
            )
            m = out.merge(agg, on=["game_id", "play_id", "nfl_id"], how="left", suffixes=("", "_last"))
            m["delta_frames"] = (m["frame_id"] - m["last_frame_id"]).clip(lower=0).astype(float)
            m["delta_t"] = m["delta_frames"] / 10.0
            return m

        if self.input_df is None or self.output_df is None:
            self.load_data(force_recompute=force_recompute)
        self.df = self._load_or_compute(
            "create_training_rows",
            _create_training_rows,
            force_recompute=force_recompute,
            input_df=self.input_df,
            output_df=self.output_df,
        )
        return self.df

    def normalize_play_direction(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Normalize play direction so that all plays move to the right.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with normalized play direction.
        """
        def _normalize_play_direction(df: pd.DataFrame) -> pd.DataFrame:
            right_df = df[df["play_direction"] == "right"].copy()
            left_df = df[df["play_direction"] == "left"].copy()

            left_df["was_left"] = 1
            right_df["was_left"] = 0

            # Reflect coordinates for left plays
            left_df["x"] = self.X_LIMIT - left_df["x"]
            left_df["y"] = self.Y_LIMIT - left_df["y"]
            left_df["ball_land_x"] = self.X_LIMIT - left_df["ball_land_x"]
            left_df["ball_land_y"] = self.Y_LIMIT - left_df["ball_land_y"]

            # Reflect target coordinates
            left_df["target_x"] = self.X_LIMIT - left_df["target_x"]
            left_df["target_y"] = self.Y_LIMIT - left_df["target_y"]

            # Adjust angles for left plays
            left_df["dir"] = (left_df["dir"] + 180) % 360
            left_df["o"] = (left_df["o"] + 180) % 360
            left_df["play_direction"] = "right"

            return pd.concat([right_df, left_df], ignore_index=True)

        if self.df is None:
            self.create_training_rows(force_recompute=force_recompute)
        self.df = self._load_or_compute(
            "normalize_play_direction",
            _normalize_play_direction,
            force_recompute=force_recompute,
            df=self.df,
        )
        return self.df

    def add_kinematic_features(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Add kinematic features: velocity components, acceleration components,
        and angle/distance to the ball.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with added kinematic features.
        """
        def _add_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            # Convert direction to radians
            df["dir_rad"] = np.deg2rad(df["dir"])

            # Velocity components
            df["vx"] = df["s"] * np.cos(df["dir_rad"])
            df["vy"] = df["s"] * np.sin(df["dir_rad"])

            # Acceleration components
            df["ax"] = df["a"] * np.cos(df["dir_rad"])
            df["ay"] = df["a"] * np.sin(df["dir_rad"])

            # Distance and angle to ball landing point
            dx_ball = df["ball_land_x"] - df["x"]
            dy_ball = df["ball_land_y"] - df["y"]
            df["dist_to_ball"] = np.sqrt(dx_ball**2 + dy_ball**2)
            df["angle_to_ball"] = np.arctan2(dy_ball, dx_ball)

            # Velocity component toward the ball
            df["v_toward_ball"] = df["vx"] * np.cos(df["angle_to_ball"]) + df["vy"] * np.sin(df["angle_to_ball"])

            return df

        if self.df is None:
            self.normalize_play_direction(force_recompute=force_recompute)
        self.df = self._load_or_compute(
            "add_kinematic_features",
            _add_kinematic_features,
            force_recompute=force_recompute,
            df=self.df,
        )
        return self.df

    def add_spatial_features(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Add spatial features: distance to sidelines and endzone.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with added spatial features.
        """
        def _add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            # Distance to sidelines and endzone
            df["dist_to_left_sideline"] = df["y"]
            df["dist_to_right_sideline"] = self.X_LIMIT - df["y"]
            df["dist_to_endzone"] = self.Y_LIMIT - df["x"]

            # Normalized frame index
            df["frame_norm"] = df["frame_id"] / df["num_frames_output"].clip(lower=1)

            return df

        if self.df is None:
            self.add_kinematic_features(force_recompute=force_recompute)
        self.df = self._load_or_compute(
            "add_spatial_features",
            _add_spatial_features,
            force_recompute=force_recompute,
            df=self.df,
        )
        return self.df

    def add_player_features(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Add player-specific features: height in meters and age.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with added player features.
        """
        def _add_player_features(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            # Convert height to meters
            def height_to_meters(h: str) -> float:
                try:
                    feet, inch = h.split("-")
                    return (int(feet) * 12 + int(inch)) * 0.0254
                except:
                    return np.nan

            df["height_m"] = df["player_height"].apply(height_to_meters)

            # Calculate age
            birth_dates = pd.to_datetime(df["player_birth_date"], errors="coerce")
            df["age_years"] = (pd.Timestamp("2025-01-01") - birth_dates).dt.days / 365.25

            return df

        if self.df is None:
            self.add_spatial_features(force_recompute=force_recompute)
        self.df = self._load_or_compute(
            "add_player_features",
            _add_player_features,
            force_recompute=force_recompute,
            df=self.df,
        )
        return self.df

    def add_team_context_features(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Add team context features: mean speed, acceleration, and distance to ball
        aggregated by game_id, play_id, frame_id, and player_side.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with added team context features.
        """
        def _add_team_context_features(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            group_keys = ["game_id", "play_id", "frame_id", "player_side"]
            agg_df = (
                df.groupby(group_keys)
                .agg(
                    team_mean_s=("s", "mean"),
                    team_std_s=("s", "std"),
                    team_mean_a=("a", "mean"),
                    team_mean_dist_to_ball=("dist_to_ball", "mean"),
                    team_count=("nfl_id", "nunique"),
                )
                .reset_index()
            )

            return df.merge(agg_df, on=group_keys, how="left")

        if self.df is None:
            self.add_player_features(force_recompute=force_recompute)
        self.df = self._load_or_compute(
            "add_team_context_features",
            _add_team_context_features,
            force_recompute=force_recompute,
            df=self.df,
        )
        return self.df

    def add_nearest_defender_distance(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Add the distance to the nearest defender for each target player.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with added nearest defender distance.
        """
        def _add_nearest_defender_distance(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            weeks = sorted(df["week"].dropna().unique())

            nearest_parts = []
            for week in weeks:
                cache_path = self.cache_dir / f"nearest_defender_w{int(week):02d}.pkl"
                if not force_recompute and cache_path.exists():
                    nearest_parts.append(pd.read_pickle(cache_path))
                    continue

                week_df = df[df["week"] == week].copy()
                targets = week_df[week_df["player_to_predict"]][["game_id", "play_id", "frame_id", "nfl_id", "x", "y"]]
                defenders = week_df[week_df["player_side"] == "Defense"][["game_id", "play_id", "frame_id", "x", "y"]]

                if targets.empty or defenders.empty:
                    nearest_parts.append(pd.DataFrame())
                    continue

                key_cols = ["game_id", "play_id", "frame_id"]
                defenders_grouped = defenders.groupby(key_cols)
                targets_grouped = targets.groupby(key_cols)

                nearest_chunk_list = []
                for key, target_group in targets_grouped:
                    try:
                        defender_group = defenders_grouped.get_group(key)
                    except KeyError:
                        tmp = target_group[["game_id", "play_id", "frame_id", "nfl_id"]].copy()
                        tmp["nearest_defender_dist"] = np.nan
                        nearest_chunk_list.append(tmp)
                        continue

                    if len(defender_group) == 0 or len(target_group) == 0:
                        tmp = target_group[["game_id", "play_id", "frame_id", "nfl_id"]].copy()
                        tmp["nearest_defender_dist"] = np.nan
                        nearest_chunk_list.append(tmp)
                        continue

                    target_coords = target_group[["x", "y"]].values
                    defender_coords = defender_group[["x", "y"]].values

                    tree = KDTree(defender_coords)
                    dist, _ = tree.query(target_coords, k=1)
                    tmp = target_group[["game_id", "play_id", "frame_id", "nfl_id"]].copy()
                    tmp["nearest_defender_dist"] = dist.flatten()
                    nearest_chunk_list.append(tmp)

                nearest_week = pd.concat(nearest_chunk_list, ignore_index=True)
                joblib.dump(nearest_week, cache_path)
                nearest_parts.append(nearest_week)

            nearest_all = pd.concat(nearest_parts, ignore_index=True)
            return df.merge(nearest_all, on=["game_id", "play_id", "frame_id", "nfl_id"], how="left")

        if self.df is None:
            self.add_team_context_features(force_recompute=force_recompute)
        self.df = self._load_or_compute(
            "add_nearest_defender_distance",
            _add_nearest_defender_distance,
            force_recompute=force_recompute,
            df=self.df,
        )
        return self.df

    def add_defensive_pressure(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Add mean distance of defenders to the ball as a measure of defensive pressure.
        This feature is added only for target players (player_to_predict=True).

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with added defensive pressure feature.
        """
        def _add_defensive_pressure(df: pd.DataFrame) -> pd.DataFrame:
            # Calculate distance to ball for defenders
            defenders = df[df["player_side"] == "Defense"].copy()
            defenders["dist_to_ball_def"] = np.sqrt(
                (defenders["ball_land_x"] - defenders["x"])**2 +
                (defenders["ball_land_y"] - defenders["y"])**2
            )

            # Aggregate mean distance to ball per frame
            defensive_pressure = (
                defenders.groupby(["game_id", "play_id", "frame_id"])
                .agg(def_mean_dist_to_ball=("dist_to_ball_def", "mean"))
                .reset_index()
            )

            # Merge with target players only
            targets = df[df["player_to_predict"]].copy()
            return targets.merge(
                defensive_pressure,
                on=["game_id", "play_id", "frame_id"],
                how="left"
            )

        if self.df is None:
            self.add_nearest_defender_distance(force_recompute=force_recompute)

        # Apply transformation and update self.df with targets only
        targets_with_pressure = self._load_or_compute(
            "add_defensive_pressure",
            _add_defensive_pressure,
            force_recompute=force_recompute,
            df=self.df,
        )

        # Merge targets back into the full DataFrame
        self.df = pd.concat([
            self.df[~self.df["player_to_predict"]],  # Non-target players
            targets_with_pressure  # Target players with new feature
        ], ignore_index=True)

        return self.df

    def add_advanced_features(self, force_recompute: bool = False) -> pd.DataFrame:
        """
        Add advanced features to the DataFrame.

        Args:
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: DataFrame with added advanced features.
        """
        def _add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()

            # Check for required features
            required_features = {
                "nearest_defender_dist": None,
                "dist_to_ball": None,
                "v_toward_ball": None,
            }

            for feature in required_features.keys():
                if feature not in df.columns:
                    print(f"Warning: Feature '{feature}' not found. Skipping related calculations.")
                    required_features[feature] = False
                else:
                    required_features[feature] = True

            # Filter stationary players only if 's' exists
            if "s" in df.columns:
                df = df[df["s"] > 0.3].copy()
            else:
                print("Warning: 's' column not found. Skipping speed filter.")

            # Time-to-ball and relative velocity (requires dist_to_ball and s)
            if required_features["dist_to_ball"] and "s" in df.columns:
                eps = 1e-3
                df["time_to_ball"] = df["dist_to_ball"] / (df["s"] + eps)
                df["v_rel_to_ball"] = df["v_toward_ball"] / (df["s"] + eps)

                # Clip extreme values
                ttb_clip = df["time_to_ball"].quantile(0.99)
                df["time_to_ball"] = df["time_to_ball"].clip(0, ttb_clip)
                df["log_time_to_ball"] = np.log1p(df["time_to_ball"])
                df["v_rel_to_ball"] = df["v_rel_to_ball"].clip(-1.0, 1.0)
            else:
                print("Warning: Skipping time_to_ball and v_rel_to_ball calculations.")

            # Heading alignment (requires dir_rad and angle_to_ball)
            if "dir_rad" in df.columns and "angle_to_ball" in df.columns:
                df["angle_diff"] = df["dir_rad"] - df["angle_to_ball"]
                df["angle_diff"] = (df["angle_diff"] + np.pi) % (2 * np.pi) - np.pi
                df["cos_to_ball"] = np.cos(df["angle_diff"])
                df["sin_to_ball"] = np.sin(df["angle_diff"])
            else:
                print("Warning: Skipping angle_diff, cos_to_ball, and sin_to_ball calculations.")

            # Pressure index (requires nearest_defender_dist)
            if required_features["nearest_defender_dist"]:
                df["pressure_index"] = 1.0 / (df["nearest_defender_dist"].fillna(10.0) + 0.5)
                df["pressure_index"] = df["pressure_index"].clip(0, 5)
                df["pressure_index_norm"] = df["pressure_index"] / df["pressure_index"].max()
            else:
                print("Warning: Skipping pressure_index calculations.")

            # Normalized coordinates
            if "x" in df.columns and "y" in df.columns:
                df["x_norm"] = df["x"] / self.X_LIMIT
                df["y_norm"] = df["y"] / self.Y_LIMIT
            else:
                print("Warning: Skipping x_norm and y_norm calculations.")

            # Forward progress indicators
            if "vx" in df.columns:
                df["is_moving_forward"] = (df["vx"] > 0).astype(np.int8)
                df["forward_speed"] = df["vx"].clip(lower=0.0)
            else:
                print("Warning: Skipping is_moving_forward and forward_speed calculations.")

            return df

        if self.df is None:
            print("Warning: No data loaded. Running minimal pipeline...")
            self.load_data(force_recompute=force_recompute)
            self.create_training_rows(force_recompute=force_recompute)
            self.normalize_play_direction(force_recompute=force_recompute)
            self.add_kinematic_features(force_recompute=force_recompute)

        self.df = self._load_or_compute(
            "add_advanced_features",
            _add_advanced_features,
            force_recompute=force_recompute,
            df=self.df,
        )
        return self.df

    def get_feature_cols(self, exclude_cols: Optional[List[str]] = None) -> List[str]:
        """
        Get the list of feature columns, excluding unnecessary ones.

        Args:
            exclude_cols (Optional[List[str]]): List of columns to exclude.

        Returns:
            List[str]: List of feature columns.
        """
        if exclude_cols is None:
            exclude_cols = [
                "player_name", "player_position", "player_role", "player_side",
                "play_direction", "player_height", "player_birth_date",
                "game_id", "play_id", "nfl_id", "id", "target_x", "target_y",
                "last_frame_id", "delta_frames", "delta_t",
            ]

        if self.df is None:
            raise ValueError("No data loaded. Run get_final_data() first.")

        # Check existing columns
        existing_cols = self.df.columns.tolist()
        exclude_cols = [col for col in exclude_cols if col in existing_cols]

        return [col for col in existing_cols if col not in exclude_cols]

    def get_final_data(
        self,
        steps: Optional[List[str]] = None,
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Get the final DataFrame with selected feature engineering steps.

        Args:
            steps (Optional[List[str]]): List of steps to execute. If None, run all steps.
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            pd.DataFrame: Final DataFrame with selected features.
        """
        # Default steps in order
        all_steps = [
            "load_data",
            "create_training_rows",
            "normalize_play_direction",
            "add_kinematic_features",
            "add_spatial_features",
            "add_player_features",
            "add_team_context_features",
            "add_nearest_defender_distance",
            "add_defensive_pressure",
            "add_advanced_features",
        ]

        # Use provided steps or all steps
        steps_to_run = steps if steps is not None else all_steps

        # Execute selected steps
        for step in steps_to_run:
            if step == "load_data":
                self.load_data(force_recompute=force_recompute)
            elif step == "create_training_rows":
                self.create_training_rows(force_recompute=force_recompute)
            elif step == "normalize_play_direction":
                self.normalize_play_direction(force_recompute=force_recompute)
            elif step == "add_kinematic_features":
                self.add_kinematic_features(force_recompute=force_recompute)
            elif step == "add_spatial_features":
                self.add_spatial_features(force_recompute=force_recompute)
            elif step == "add_player_features":
                self.add_player_features(force_recompute=force_recompute)
            elif step == "add_team_context_features":
                self.add_team_context_features(force_recompute=force_recompute)
            elif step == "add_nearest_defender_distance":
                self.add_nearest_defender_distance(force_recompute=force_recompute)
            elif step == "add_defensive_pressure":
                self.add_defensive_pressure(force_recompute=force_recompute)
            elif step == "add_advanced_features":
                self.add_advanced_features(force_recompute=force_recompute)

        return self.df

    def prepare_sequential_data(
        self,
        n_timesteps_in: int = 10,
        n_timesteps_out: int = 10,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for sequential models.
        Creates input sequences of shape (n_samples, n_timesteps_in, n_features)
        and output sequences of shape (n_samples, n_timesteps_out * 2).

        Args:
            n_timesteps_in (int): Number of input timesteps (history).
            n_timesteps_out (int): Number of output timesteps (future).
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and output sequences.
        """
        def _prepare_sequential_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            X, y = [], []
            feature_cols = self.get_feature_cols()  # Get a feature_cols

            for (game_id, play_id, nfl_id), group in df.groupby(["game_id", "play_id", "nfl_id"]):
                group = group.sort_values("frame_id")

                # Check if the group has enough frames
                if len(group) < n_timesteps_in + n_timesteps_out:
                    print(f"Skipping {game_id}_{play_id}_{nfl_id}: only {len(group)} frames")
                    continue

                # Input: last `n_timesteps_in` frames before the throw
                input_seq = group.iloc[-n_timesteps_in - n_timesteps_out:-n_timesteps_out][feature_cols].values
                if len(input_seq) < n_timesteps_in:
                    continue

                # Output: next `n_timesteps_out` pairs of (x, y)
                output_seq = group.iloc[-n_timesteps_out:][["target_x", "target_y"]].values.flatten()

                X.append(input_seq)
                y.append(output_seq)

            return np.array(X), np.array(y)

        if self.df is None:
            self.get_final_data(force_recompute=force_recompute)

        return self._load_or_compute(
            f"prepare_sequential_data_{n_timesteps_in}_{n_timesteps_out}",
            _prepare_sequential_data,
            force_recompute=force_recompute,
            df=self.df,
        )
    
    def prepare_sequential_data_dynamic(
        self,
        n_timesteps_in: int = 5,
        force_recompute: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Prepare sequential data with dynamic output length for each player.
        For each player, n_timesteps_out = n_frames - n_timesteps_in.

        Args:
            n_timesteps_in (int): Number of input timesteps (history).
            force_recompute (bool): If True, recompute even if cached.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]:
                - List of input sequences (shape: [n_timesteps_in, n_features]).
                - List of output sequences (shape: [n_timesteps_out * 2]).
        """
        def _prepare_sequential_data(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
            X, y = [], []
            feature_cols = self.get_feature_cols()

            for (game_id, play_id, nfl_id), group in df.groupby(["game_id", "play_id", "nfl_id"]):
                group = group.sort_values("frame_id")
                n_frames = len(group)

                # Skip groups with insufficient frames
                if n_frames <= n_timesteps_in:
                    print(f"Skipping {game_id}_{play_id}_{nfl_id}: only {n_frames} frames")
                    continue

                # Dynamic n_timesteps_out for this player
                n_timesteps_out = n_frames - n_timesteps_in
                print(f"Player {game_id}_{play_id}_{nfl_id}: {n_frames} frames → "
                    f"n_timesteps_in={n_timesteps_in}, n_timesteps_out={n_timesteps_out}")

                # Input: first n_timesteps_in frames
                input_seq = group.iloc[:n_timesteps_in][feature_cols].values

                # Output: next n_timesteps_out (x, y) pairs
                output_seq = group.iloc[n_timesteps_in:n_timesteps_in + n_timesteps_out][["target_x", "target_y"]].values.flatten()

                X.append(input_seq)
                y.append(output_seq)

            return X, y

        if self.df is None:
            self.get_final_data(force_recompute=force_recompute)

        return self._load_or_compute(
            f"prepare_sequential_data_dynamic_{n_timesteps_in}",
            _prepare_sequential_data,
            force_recompute=force_recompute,
            df=self.df,
        )


class ReverseTransformer:
    """
    A class to reverse transformations applied during feature engineering.
    Handles coordinate reflections, normalizations, and other adjustments.
    """

    def __init__(self, x_limit: float = 120.0, y_limit: float = 53.3):
        """
        Initialize with field dimensions.

        Args:
            x_limit (float): Maximum x-coordinate (field length).
            y_limit (float): Maximum y-coordinate (field width).
        """
        self.X_LIMIT = x_limit
        self.Y_LIMIT = y_limit

    def reverse_coordinates(
        self,
        df: pd.DataFrame,
        was_left_col: str = "was_left",
        x_col: str = "x",
        y_col: str = "y",
        target_x_col: str = "target_x",
        target_y_col: str = "target_y",
    ) -> pd.DataFrame:
        """
        Reverse coordinate reflections for plays originally moving left.

        Args:
            df (pd.DataFrame): DataFrame with predictions.
            was_left_col (str): Column indicating if the play was reflected.
            x_col (str): Column with x-coordinate predictions.
            y_col (str): Column with y-coordinate predictions.
            target_x_col (str): Column with target x-coordinates (if any).
            target_y_col (str): Column with target y-coordinates (if any).

        Returns:
            pd.DataFrame: DataFrame with reversed coordinates.
        """
        df = df.copy()
        mask = df[was_left_col] == 1

        # Reverse predicted coordinates
        if x_col in df.columns:
            df.loc[mask, x_col] = self.X_LIMIT - df.loc[mask, x_col]
        if y_col in df.columns:
            df.loc[mask, y_col] = self.Y_LIMIT - df.loc[mask, y_col]

        # Reverse target coordinates if present
        if target_x_col in df.columns:
            df.loc[mask, target_x_col] = self.X_LIMIT - df.loc[mask, target_x_col]
        if target_y_col in df.columns:
            df.loc[mask, target_y_col] = self.Y_LIMIT - df.loc[mask, target_y_col]

        return df

    def reverse_angles(
        self,
        df: pd.DataFrame,
        was_left_col: str = "was_left",
        dir_col: str = "dir",
        o_col: str = "o",
    ) -> pd.DataFrame:
        """
        Reverse angle adjustments for plays originally moving left.

        Args:
            df (pd.DataFrame): DataFrame with predictions.
            was_left_col (str): Column indicating if the play was reflected.
            dir_col (str): Column with direction angles.
            o_col (str): Column with orientation angles.

        Returns:
            pd.DataFrame: DataFrame with reversed angles.
        """
        df = df.copy()
        mask = df[was_left_col] == 1

        if dir_col in df.columns:
            df.loc[mask, dir_col] = (df.loc[mask, dir_col] + 180) % 360
        if o_col in df.columns:
            df.loc[mask, o_col] = (df.loc[mask, o_col] + 180) % 360

        return df

    def denormalize_coordinates(
        self,
        df: pd.DataFrame,
        x_col: str = "x_norm",
        y_col: str = "y_norm",
        x_denorm: str = "x",
        y_denorm: str = "y",
    ) -> pd.DataFrame:
        """
        Denormalize coordinates from [0, 1] to original scale.

        Args:
            df (pd.DataFrame): DataFrame with normalized coordinates.
            x_col (str): Column with normalized x-coordinates.
            y_col (str): Column with normalized y-coordinates.
            x_denorm (str): Column to store denormalized x-coordinates.
            y_denorm (str): Column to store denormalized y-coordinates.

        Returns:
            pd.DataFrame: DataFrame with denormalized coordinates.
        """
        df = df.copy()
        if x_col in df.columns:
            df[x_denorm] = df[x_col] * self.X_LIMIT
        if y_col in df.columns:
            df[y_denorm] = df[y_col] * self.Y_LIMIT
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all reverse transformations to a DataFrame with predictions.

        Args:
            df (pd.DataFrame): DataFrame with predictions.

        Returns:
            pd.DataFrame: DataFrame with reversed transformations.
        """
        df = self.reverse_coordinates(df)
        df = self.reverse_angles(df)
        df = self.denormalize_coordinates(df)
        return df


class SequentialPredictor:
    """
    A class to make sequential predictions using a trained model.
    Supports autoregressive prediction and reverse transformations.
    """

    def __init__(
        self,
        model,
        feature_cols: List[str],
        reverse_transformer: Optional[ReverseTransformer] = None,
        x_limit: float = 120.0,
        y_limit: float = 53.3,
    ):
        """
        Initialize with a trained model and feature columns.

        Args:
            model: Trained model (must have a `predict` method).
            feature_cols (List[str]): List of feature columns used for prediction.
            reverse_transformer (Optional[ReverseTransformer]): Transformer for reversing coordinate adjustments.
            x_limit (float): Maximum x-coordinate (field length).
            y_limit (float): Maximum y-coordinate (field width).
        """
        self.model = model
        self.feature_cols = feature_cols
        self.reverse_transformer = reverse_transformer or ReverseTransformer(x_limit, y_limit)
        self.X_LIMIT = x_limit
        self.Y_LIMIT = y_limit
        self.predictions = []
        self.inputs = []

    def create_next_input(
        self,
        current_input: pd.Series,
        prediction: np.ndarray,
        delta_t: float = 0.1,
    ) -> pd.Series:
        """
        Create the next input DataFrame row from the current input and prediction.

        Args:
            current_input (pd.Series): Current input row.
            prediction (np.ndarray): Predicted [x, y] coordinates.
            delta_t (float): Time step between frames (default: 0.1 seconds).

        Returns:
            pd.Series: Next input row with updated features.
        """
        next_input = current_input.copy()
        x_prev, y_prev = current_input["x"], current_input["y"]
        x_pred, y_pred = prediction[0], prediction[1]

        # Update coordinates
        next_input["x"] = x_pred
        next_input["y"] = y_pred

        # Calculate velocity and acceleration
        dx = x_pred - x_prev
        dy = y_pred - y_prev
        next_input["s"] = np.sqrt(dx**2 + dy**2) / delta_t
        next_input["a"] = (next_input["s"] - current_input["s"]) / delta_t

        # Update orientation (angle of movement)
        next_input["o"] = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        # Increment frame_id
        next_input["frame_id"] += 1

        return next_input

    def add_lag_features(self, df: pd.DataFrame, n_lags: int = 2) -> pd.DataFrame:
        """
        Add lag features to the DataFrame for autoregressive prediction.

        Args:
            df (pd.DataFrame): DataFrame with sequential data.
            n_lags (int): Number of lag steps to add (default: 2).

        Returns:
            pd.DataFrame: DataFrame with lag features.
        """
        df = df.copy()
        group_cols = ["game_id", "play_id", "nfl_id"]
        for lag in range(1, n_lags + 1):
            for col in ["x", "y", "s", "a", "vx", "vy"]:
                df[f"{col}_lag{lag}"] = df.groupby(group_cols)[col].shift(lag)
        return df

    def predict_sequence(
        self,
        initial_input: pd.DataFrame,
        n_steps: int,
        target_cols: List[str] = ["target_x", "target_y"],
        return_inputs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[pd.DataFrame]]]:
        """
        Make sequential predictions for `n_steps` steps ahead.

        Args:
            initial_input (pd.DataFrame): Initial input DataFrame (single row).
            n_steps (int): Number of steps to predict ahead.
            target_cols (List[str]): List of target columns (default: ["target_x", "target_y"]).
            return_inputs (bool): If True, return intermediate inputs (default: False).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, List[pd.DataFrame]]]:
                - Predicted sequence as a numpy array (shape: [n_steps, len(target_cols)]).
                - If `return_inputs=True`, also returns a list of intermediate inputs.
        """
        self.predictions = []
        self.inputs = [initial_input.copy()]

        for _ in range(n_steps):
            # Prepare current input (add lag features if needed)
            current_input = self.add_lag_features(self.inputs[-1]).iloc[[-1]]

            # Predict next step
            X = current_input[self.feature_cols].values
            pred = self.model.predict(X.reshape(1, -1))[0]
            self.predictions.append(pred)

            # Create next input
            next_input = self.create_next_input(current_input.iloc[0], pred)
            self.inputs.append(next_input)

        predictions_array = np.array(self.predictions)
        if return_inputs:
            return predictions_array, self.inputs
        return predictions_array

    def predict_and_reverse_transform(
        self,
        initial_input: pd.DataFrame,
        n_steps: int,
        target_cols: List[str] = ["target_x", "target_y"],
    ) -> np.ndarray:
        """
        Make sequential predictions and reverse transformations.

        Args:
            initial_input (pd.DataFrame): Initial input DataFrame (single row).
            n_steps (int): Number of steps to predict ahead.
            target_cols (List[str]): List of target columns (default: ["target_x", "target_y"]).

        Returns:
            np.ndarray: Predicted sequence with reversed transformations (original scale).
        """
        # Make predictions in transformed space
        predictions = self.predict_sequence(initial_input, n_steps, target_cols)

        # Create a DataFrame for reverse transformation
        pred_df = pd.DataFrame(predictions, columns=target_cols)
        pred_df["was_left"] = initial_input["was_left"].iloc[0]
        pred_df = self.reverse_transformer.transform(pred_df)

        return pred_df[target_cols].values
