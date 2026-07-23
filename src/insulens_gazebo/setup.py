from glob import glob
from setuptools import find_packages, setup


package_name = "insulens_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
        ("share/" + package_name + "/models/insulator", glob("models/insulator/*")),
        (
            "share/" + package_name + "/models/inspection_drone",
            glob("models/inspection_drone/*"),
        ),
        (
            "share/" + package_name + "/models/transmission_tower",
            glob("models/transmission_tower/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="insulens contributors",
    maintainer_email="maintainers@example.com",
    description="Gazebo simulation and synthetic data generation for insulens.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "generate_dataset = insulens_gazebo.dataset_generator:main",
            "patrol = insulens_gazebo.patrol:main",
        ],
    },
)
