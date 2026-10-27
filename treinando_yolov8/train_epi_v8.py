from ultralytics import YOLO

# para marcar as imagens
# https://www.makesense.ai/


def main():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Use the model
    model.train(
        data="epi.yaml",
        epochs=10,
        workers=0,
        batch=4,
        imgsz=512
    )
    metrics = model.val(batch=9)  # evaluate model performance on the validation set


if __name__ == '__main__':
    main()
