from glob import glob
from setuptools import find_packages, setup


package_name = "doggo_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="doggo contributors",
    maintainer_email="maintainers@example.com",
    description="YOLOv10 ROS 2 insulator detector and training entry point.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector = doggo_perception.detector_node:main",
            "image_source = doggo_perception.image_source:main",
            "inspection_monitor = doggo_perception.inspection_monitor:main",
            "train_yolov10 = doggo_perception.train:main",
        ],
    },
)
