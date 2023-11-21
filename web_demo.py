import streamlit as st

import config
from infer import InferClassify


@st.cache_resource
def get_model():
    args = config.Args().get_parser()
    model = InferClassify(args)
    # st.success("Loaded NLP model from Hugging Face!")
    return model


def predict(input):
    model = get_model()
    res = model.predict(input)

    return res


# create a prompt text for the text generation
prompt_text = st.text_area(label="文本分类", height=100, placeholder="请在这儿输入分类文本")


if st.button("确认", key="predict"):
    with st.spinner("请稍等........"):
        st.text('分类结果：')
        st.markdown("{}".format(predict(prompt_text)[0]))
