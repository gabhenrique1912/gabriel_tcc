from ultralytics import YOLO

# para marcar as imagens
# https://www.makesense.ai/


def main():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Use the model
    model.train(data="epi.yaml", epochs=30)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print("path", path)


if __name__ == '__main__':
    # freeze_support()
    main()
