from dataclasses import dataclass
from typing import List

import tomllib


@dataclass
class CameraConfig:
    exposure: float
    kernel_size: List[int]
    fps: float
    num_of_images: int


@dataclass
class MotorConfig:
    controllername: str
    stages: List[str]
    refmodes: List[str]
    serialnum: str


@dataclass
class AxesConfig:
    x: int
    y: int
    z: int


@dataclass
class VertexConfig:
    pt1: List[float]
    pt2: List[float]


@dataclass
class MovementConfig:
    dx: float
    dy: float
    dz: float
    min_z: float
    num_steps_x: int
    num_steps_y: int


@dataclass
class FileConfig:
    save_dir: str
    model_path: str


@dataclass
class Config:
    camera: CameraConfig
    motor: MotorConfig
    axes: AxesConfig
    vertex: VertexConfig
    movement: MovementConfig
    file: FileConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        return cls(
            camera=CameraConfig(**config_dict["CAMERA"]),
            motor=MotorConfig(**config_dict["MOTOR"]),
            axes=AxesConfig(**config_dict["AXES"]),
            vertex=VertexConfig(**config_dict["VERTEX"]),
            movement=MovementConfig(**config_dict["MOVEMENT"]),
            file=FileConfig(**config_dict["FILE"]),
        )


def load_config(config_path="config.toml"):
    with open(config_path, "rb") as file:
        config = tomllib.load(file)

    config["MOVEMENT"]["num_steps_x"] = int(
        abs(config["VERTEX"]["pt1"][0] - config["VERTEX"]["pt2"][0])
        // config["MOVEMENT"]["dx"]
    )

    config["MOVEMENT"]["num_steps_y"] = int(
        abs(config["VERTEX"]["pt1"][1] - config["VERTEX"]["pt2"][1])
        // config["MOVEMENT"]["dy"]
    )

    return Config.from_dict(config)
