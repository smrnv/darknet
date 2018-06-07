import uuid
from os import remove

import _darknet as darknet
from flask import Flask, request, Request, jsonify
from werkzeug.datastructures import FileStorage

app = Flask("__name__")

configPath = "cfg/nfpa.cfg"
weightPath = "nfpa_19900_gpu.weights"
metaPath = "cfg/nfpa.data"
tempDir = "temp"
threshold = 0.25


@app.route("/", methods=['POST'])
def perform():
    savedFile = saveFile(request)
    result = detectAndDeleteFile(savedFile)

    return jsonify(result.__dict__)


def saveFile(req: Request) -> str:
    # TODO: check form field existence and image type
    random_name = str(uuid.uuid4())
    form_field: FileStorage = req.files['image']
    extension = str(form_field.content_type).replace("image/", ".")
    path = tempDir + "/" + random_name + extension
    with open(path, "wb") as file:
        form_field.save(file)

    return path


class Detection:
    def __init__(self, object_class: str, confidence: float,
                 center_x: float, center_y: float, width: float, height: float):
        self.object_class = object_class
        self.confidence = confidence
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height


def detectAndDeleteFile(file_path: str) -> Detection:
    try:
        return detect(file_path)
    finally:
        remove(file_path)


def detect(file_path: str) -> Detection:
    result = darknet.performDetect(imagePath=file_path,
                                   thresh=threshold,
                                   configPath=configPath,
                                   weightPath=weightPath,
                                   metaPath=metaPath,
                                   showImage=False,
                                   makeImageOnly=False,
                                   initOnly=False)
    return Detection(
        object_class=result[0][0],
        confidence=result[0][1],
        center_x=result[0][2][0],
        center_y=result[0][2][1],
        width=result[0][2][2],
        height=result[0][2][3]
    )


if __name__ == "__main__":
    result = detect("data/fire_diamond/lego_diamond.jpg")
    print(result.__dict__)
