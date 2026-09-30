from roboflow import Roboflow

rf = Roboflow(api_key="7EdWUUZ4KVSWAQSdLDzn")
project = rf.workspace("raghav-bharathi-vywgh").project("rubik-s-cube-ivvrq")
version = project.version(3)
dataset = version.download("coco-segmentation
