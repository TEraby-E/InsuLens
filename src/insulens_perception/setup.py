from glob import glob
from setuptools import find_packages, setup


package_name = "insulens_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/models", glob("models/*.yaml")),
        ("share/" + package_name + "/web/dashboard", glob("web/dashboard/*")),
        ("share/" + package_name + "/web/inspection_portal", glob("web/inspection_portal/*")),
    ],
    install_requires=["setuptools", "imageio-ffmpeg>=0.5,<1.0"],
    zip_safe=True,
    maintainer="insulens contributors",
    maintainer_email="maintainers@example.com",
    description="YOLOv10 ROS 2 insulator detector and training entry point.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector = insulens_perception.detector_node:main",
            "image_source = insulens_perception.image_source:main",
            "inspection_monitor = insulens_perception.inspection_monitor:main",
            "train_yolov10 = insulens_perception.train:main",
            "analyze_small_objects = insulens_perception.analyze_dataset:main",
            "insulens_web = insulens_perception.web_app:main",
            "optimize_insulens_model = insulens_perception.optimize_model:main",
        ],
    },
)
