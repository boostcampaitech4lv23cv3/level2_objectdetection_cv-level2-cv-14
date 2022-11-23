import streamlit as st
import numpy as np
from PIL import Image
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO


@st.cache
def load_model(config, checkpint, device):
    return init_detector(config, checkpint, device=device)


def show_train_img(img, *classes):
    coco = COCO("../dataset/train.json")
    img_id = img.split(".")[0]
    cat = [idx for idx, value in enumerate(classes) if value]
    annotation_ids = coco.getAnnIds(imgIds=int(img_id), catIds=cat)
    anns = coco.loadAnns(annotation_ids)
    im = Image.open("../dataset/train/" + img)

    fig, ax = plt.subplots()
    for ann in anns:
        box = ann["bbox"]
        bb = patches.Rectangle(
            (box[0], box[1]),
            box[2],
            box[3],
            linewidth=2,
            edgecolor=st.session_state.color[ann["category_id"]],
            facecolor="none",
        )
        ax.add_patch(bb)
    ax.imshow(im)
    plt.axis("off")
    st.pyplot(fig)


def show_prediction(model, img):
    img = mmcv.imread("../dataset/test/" + img)
    result = inference_detector(model, img)
    im = Image.fromarray(model.show_result(img, result))
    st.image(im, caption="Detected Image")


def main():
    st.title("재활용 품목 분류를 위한 Object Detectionl")

    config_file = "/opt/ml/baseline/Experiment/yolov3/yolo.py"
    checkpoint_file = "/opt/ml/baseline/work_dirs/swin/best.pt"
    device = "cuda"

    if "model" not in st.session_state:
        st.session_state.model = load_model(config_file, checkpoint_file, device)

    with st.sidebar:
        General_trash = st.checkbox("General_trash", value=False)
        Paper = st.checkbox("Paper", value=False)
        Paper_pack = st.checkbox("Paper_pack", value=False)
        Metal = st.checkbox("Metal", value=False)
        Glass = st.checkbox("Glass", value=False)
        Plastic = st.checkbox("Plastic", value=False)
        Styrofoam = st.checkbox("Styrofoam", value=False)
        Plastic_bag = st.checkbox("Plastic_bag", value=False)
        Battery = st.checkbox("Battery", value=False)
        Clothing = st.checkbox("Clothing", value=False)

    st.session_state.color = [
        (1, 0, 0),  # Red
        (1, 0.5, 0),  # Orange
        (1, 1, 0),  # Yellow
        (0.5, 1, 0),  # Green
        (0, 1, 1),  # Sky blue
        (0, 0.5, 1),  # Middle blue
        (0, 0, 1),  # Blue
        (0.5, 0, 1),  # Purple
        (1, 0, 1),  # Pink
        (0.33, 0.33, 0.33),  # Gray
    ]

    tab1, tab2 = st.tabs(["Train_img_View", "Test_img_View"])
    with tab1:
        train_img = st.selectbox(
            "Choose Train img", sorted(os.listdir("../dataset/train/"))
        )
        if train_img:
            show_train_img(
                train_img,
                General_trash,
                Paper,
                Paper_pack,
                Metal,
                Glass,
                Plastic,
                Styrofoam,
                Plastic_bag,
                Battery,
                Clothing,
            )

    with tab2:
        test_img = st.selectbox(
            "Choose Test img", sorted(os.listdir("../dataset/test/"))
        )
        if test_img:
            show_prediction(st.session_state.model, test_img)


if __name__ == "__main__":
    main()
