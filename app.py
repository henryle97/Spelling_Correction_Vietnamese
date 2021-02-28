import time
import streamlit as st
import SessionState
import numpy as np
from predictor import Predictor
from dataset.add_noise import SynthesizeData

state = SessionState.get(text_correct="", input="")
import nltk

def main():
    model, synther = load_model()
    st.title("Chương trình sửa lỗi chính tả tiếng việt")
    # Load model
    state.input = ""
    text_input = st.text_area("Gõ câu sai tại đây:", value=state.input)
    if st.button("Sinh lỗi"):
        noise_text = synther.add_noise(text_input, percent_err=0.15)
        state.output = noise_text
        text_input = st.text_area("Câu sai sinh:", value=state.output)
    if st.button("Correct"):
        text_input = model.spelling_correct(text_input)

        state.text_correct = model.spelling_correct(text_input)


        st.text("Câu nhiễu: ")
        st.success(text_input)
        st.text("Kết quả:")
        st.success(state.text_correct)

    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model():
    print("Loading model ...")
    nltk.download('punkt')
    model = Predictor(weight_path='weights/seq2seq.pth')
    synther = SynthesizeData()
    return model, synther


if __name__ == "__main__":
    main()