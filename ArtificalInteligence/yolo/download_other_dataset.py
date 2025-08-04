from roboflow import Roboflow

rf = Roboflow(api_key="NymhsACx0zBHsU5yYCrl")
project = rf.workspace("nibm-7v215").project("dent-xehao")
dataset = project.version(1).download("yolov8")


# 243
# https://universe.roboflow.com/abdul-abdul-csegs/proj-u3e0k/dataset/1#
rf = Roboflow(api_key="NymhsACx0zBHsU5yYCrl")
project = rf.workspace("abdul-abdul-csegs").project("proj-u3e0k")
dataset = project.version(1).download("yolov8")


# 1042 images
# https://universe.roboflow.com/dhruv-kepml/dent-detection-hy5zl/dataset/1#
rf = Roboflow(api_key="NymhsACx0zBHsU5yYCrl")
project = rf.workspace("dhruv-kepml").project("dent-detection-hy5zl")
dataset = project.version(1).download("yolov8")


#3001
# https://universe.roboflow.com/car-damage-kadad/car-damage-v5/dataset/6#
rf = Roboflow(api_key="NymhsACx0zBHsU5yYCrl")
project = rf.workspace("car-damage-kadad").project("car-damage-v5")
dataset = project.version(6).download("yolov8")