import streamlit as st

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗"
)

st.title("🚗 Driver Drowsiness Detection using Deep Learning")

st.write(
    "Upload a driver's facial image to detect drowsiness "
    "using a deep learning model."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")