import time
import streamlit as st
import SessionState
import numpy as np
from predictor import Predictor

state = SessionState.get(text_correct="")


def main():
    model = load_model()
    st.title("Chương trình sửa lỗi chính tả tiếng việt")
    # Load model

    text_input = st.text_area("Gõ câu sai tại đây:")
    if st.button("Correct"):
        state.text_correct = model.spelling_correct(text_input)

    st.text("Kết quả:")
    st.success(state.text_correct)
    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model():
    print("Loading model ...")
    model = Predictor(weight_path='weights/seq2seq.pth')
    return model


if __name__ == "__main__":
    main()