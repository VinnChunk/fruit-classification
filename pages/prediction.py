import streamlit as st
import tempfile
import os

from config import IMG_SIZE_CLASSIFICATION
from config import AUGMENT_PATH

from utils.classifier import (
    get_model,
    predict_image,
    split_label
)


def show_prediction():

    st.title("🔍 Fruit Prediction")

    st.write(
        "Upload gambar buah untuk mengetahui apakah buah tersebut Fresh atau Rotten."
    )

    model_name = st.selectbox(
        "Pilih Model",
        [
            "KNN",
            "SVM",
            "ANN"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    st.image(
        uploaded_file,
        width=300
    )

    if st.button("🔍 Predict"):

        # Pastikan dataset augmentation sudah ada
        if not os.path.exists(AUGMENT_PATH):

            st.error(
                "Dataset hasil augmentation belum tersedia. Jalankan proses Augmentation terlebih dahulu."
            )

            return

        # Simpan gambar upload sementara
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            tmp.write(uploaded_file.read())
            image_path = tmp.name

        try:

            with st.spinner("Menyiapkan model..."):

                model, encoder = get_model(
                    model_name,
                    AUGMENT_PATH,
                    IMG_SIZE_CLASSIFICATION
                )

            label = predict_image(
                model,
                encoder,
                image_path,
                IMG_SIZE_CLASSIFICATION
            )

            fruit, condition = split_label(label)

            st.success("Prediction Completed")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fruit",
                    fruit
                )

            with col2:

                if condition == "Fresh":
                    st.success("🟢 Fresh")

                else:
                    st.error("🔴 Rotten")

        finally:

            # Hapus file sementara
            if os.path.exists(image_path):
                os.remove(image_path)