import streamlit as st
import numpy as np
from PIL import Image
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pathlib import Path
import random


@st.cache(allow_output_mutation=True)
def load_model(config, checkpint, device):
    return init_detector(config, checkpint, device=device)


@st.cache(allow_output_mutation=True)
def load_coco(path):
    return COCO(path)


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def show_train_img(img, coco, *classes):
    img_id = img.split(".")[0]
    cat = [idx for idx, value in enumerate(classes) if value]
    annotation_ids = coco.getAnnIds(imgIds=int(img_id), catIds=cat)
    anns = coco.loadAnns(annotation_ids)
    im = Image.open("../dataset/train/" + img)

    fig, ax = plt.subplots()
    plt.axis("off")
    for ann in anns:
        box = ann["bbox"]
        b1 = patches.Rectangle(
            (box[0], box[1]),
            box[2],
            box[3],
            linewidth=2,
            edgecolor=st.session_state.color[ann["category_id"]],
            facecolor="none",
        )
        ax.add_patch(b1)
    ax.imshow(im)
    st.pyplot(fig)


def show_prediction(model, select, img, *classes):
    fig, ax = plt.subplots()
    plt.axis("off")
    im = mmcv.imread(f"../dataset/{select}/" + img)
    result = inference_detector(model, im)
    cat = [idx for idx, value in enumerate(classes) if value]
    for c in cat:
        boxes = result[c][result[c][:, 4] > 0.3][:, :-1]
        for box in boxes:
            b2 = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=st.session_state.color[c],
                facecolor="none",
            )
            ax.add_patch(b2)
    im = Image.open(f"../dataset/{select}/" + img)
    ax.imshow(im)
    st.pyplot(fig)


def main(config_file, checkpoint_file, train_json, val_json, test_path):
    """
    Args:
        config_file (str): 확인할 모델의 config 주소를 입력하세요
        checkpoint_file (str): 사전에 훈련된 모델의 checkpoint 주소를 입력하세요
        train_json (str): 확인할 train data의 json 주소를 입력하세요
        val_json (str): 확인할 validation data의 json 주소를 입력하세요
        test_path (str): 확인할 test data의 폴더 주소를 입력하세요
    """

    st.title("재활용 품목 분류를 위한 Object Detection")
    if "model" not in st.session_state:
        st.session_state.model = load_model(config_file, checkpoint_file, "cuda")
    if "train_coco" not in st.session_state:
        st.session_state.train_coco = COCO(train_json)
    if "valid_coco" not in st.session_state:
        st.session_state.valid_coco = COCO(val_json)
    if "color" not in st.session_state:
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

    with st.sidebar:
        st.write("확인하고 싶은 bbox의 class를 선택하세요.")
        General_trash = st.checkbox("General_trash", value=True)
        Paper = st.checkbox("Paper", value=True)
        Paper_pack = st.checkbox("Paper_pack", value=True)
        Metal = st.checkbox("Metal", value=True)
        Glass = st.checkbox("Glass", value=True)
        Plastic = st.checkbox("Plastic", value=True)
        Styrofoam = st.checkbox("Styrofoam", value=True)
        Plastic_bag = st.checkbox("Plastic_bag", value=True)
        Battery = st.checkbox("Battery", value=True)
        Clothing = st.checkbox("Clothing", value=True)

    classes = {
        "General_trash": General_trash,
        "Paper": Paper,
        "Paper_pack": Paper_pack,
        "Metal": Metal,
        "Glass": Glass,
        "Plastic": Plastic,
        "Styrofoam": Styrofoam,
        "Plastic_bag": Plastic_bag,
        "Battery": Battery,
        "Clothing": Clothing,
    }

    tab1, tab2 = st.tabs(["img_View", "Data_argument"])
    with tab1:
        img_type = st.radio(
            "Select Train or Validation or Test", ("Train", "Validation", "Test")
        )

        if img_type == "Test":
            test_img = st.selectbox("Choose Test img", sorted(os.listdir(test_path)))
            if st.button("random choice"):
                test_img = random.choice(sorted(os.listdir(test_path)))
                st.write(test_img)
            if test_img:
                show_prediction(
                    st.session_state.model, "test", test_img, *classes.values()
                )

        else:
            if img_type == "Train":
                my_coco = st.session_state.train_coco
            elif img_type == "Validation":
                my_coco = st.session_state.valid_coco
            options = st.multiselect("포함되어야 하는 class를 선택해주세요", classes.keys())
            options_dict = {i: j for i, j in zip(classes.keys(), range(10))}
            img_list = sorted(
                my_coco.getImgIds(catIds=[options_dict[opt] for opt in options])
            )
            img_box = st.selectbox("Choose img", img_list)

            if st.button("random choice"):
                img_box = random.choice(img_list)
                st.write(f"random choice img : {img_box}")

            img_box = str(img_box).zfill(4) + ".jpg"
            if img_box:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Ground Truth")
                    show_train_img(img_box, my_coco, *classes.values())
                with col2:
                    st.header("Predict bbox")
                    show_prediction(
                        st.session_state.model, "train", img_box, *classes.values()
                    )
        st.markdown(read_markdown_file("box_color.md"), unsafe_allow_html=True)
    with tab2:
        st.write("developping...")


if __name__ == "__main__":
    config_file = "/opt/ml/baseline/Experiment/yolov3/yolo.py"
    checkpoint_file = "/opt/ml/baseline/work_dirs/swin/best.pt"
    train_json = "../dataset/train_randomsplit_2022.json"
    val_json = "../dataset/val_randomsplit_2022.json"
    test_path = "../dataset/test/"
    main(config_file, checkpoint_file, train_json, val_json, test_path)
