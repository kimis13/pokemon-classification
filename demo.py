import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image

import pandas as pd
import streamlit as st


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet34(num_classes):
    model = models.resnet34(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


@st.cache_resource
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = build_resnet34(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model, class_names


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_top5(model, image, class_names):
    transform = get_transform()

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

        top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)

    top5_probs = top5_probs.squeeze(0).cpu().numpy()
    top5_indices = top5_indices.squeeze(0).cpu().numpy()

    results = []

    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
        results.append({
            "Rank": rank,
            "Pokemon Class": class_names[idx],
            "Probability": prob,
            "Probability (%)": f"{prob * 100:.2f}%"
        })

    return results


st.set_page_config(
    page_title="Pokemon Top-5 Classifier",
    page_icon="⚡",
    layout="centered",
)

st.title("Pokemon Image Classifier")
st.write("포켓몬 이미지를 업로드하면 ResNet-34 모델이 가장 유사한 클래스 5개를 예측합니다.")

st.sidebar.header("Model Setting")

checkpoint_path = st.sidebar.text_input(
    "Checkpoint path",
    value="./results/layer3_layer4/best_model.pth"
)

st.sidebar.write(f"Device: `{DEVICE}`")

uploaded_file = st.file_uploader(
    "포켓몬 이미지를 업로드하세요.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Top-5 Predict"):
        try:
            model, class_names = load_model(checkpoint_path)
            results = predict_top5(model, image, class_names)

            st.subheader("Top-5 Most Similar Pokemon Classes")

            top1 = results[0]
            st.success(
                f"Top-1 Prediction: {top1['Pokemon Class']} "
                f"({top1['Probability (%)']})"
            )

            df = pd.DataFrame(results)

            st.table(
                df[["Rank", "Pokemon Class", "Probability (%)"]]
            )

            st.subheader("Confidence Scores")

            for item in results:
                st.write(
                    f"**{item['Rank']}. {item['Pokemon Class']}** "
                    f"— {item['Probability (%)']}"
                )
                st.progress(float(item["Probability"]))

        except FileNotFoundError:
            st.error(f"Checkpoint file not found: {checkpoint_path}")

        except KeyError as e:
            st.error(
                f"Checkpoint format error. Missing key: {e}\n\n"
                "checkpoint에는 `model_state_dict`, `class_names`, `num_classes`가 있어야 합니다."
            )

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("이미지를 업로드하면 Top-5 예측 결과를 확인할 수 있습니다.")