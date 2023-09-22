# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import cv2
from PIL import Image
import numpy as np


def rot_degree(img, degree):
    rows, cols = img.shape
    center = (cols / 2, rows / 2)
    mask = img.copy()
    mask[:, :] = 255
    M = cv2.getRotationMatrix2D(center, degree, 1)
    top_right = np.array((cols - 1, 0)) - np.array(center)
    bottom_right = np.array((cols - 1, rows - 1)) - np.array(center)
    top_right_after_rot = M[0:2, 0:2].dot(top_right)
    bottom_right_after_rot = M[0:2, 0:2].dot(bottom_right)
    new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
    new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
    offset_x = (new_width - cols) / 2
    offset_y = (new_height - rows) / 2
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    dst = cv2.warpAffine(img, M, (new_width, new_height))
    mask = cv2.warpAffine(mask, M, (new_width, new_height))
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return dst, mask


def run():
    st.title("Image Rotate Application")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    degree_value = st.slider('degree', min_value=0, max_value=360, value=50)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Processing...")

        result_img, _ = rot_degree(np.array(image), degree_value)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Input Image")
            st.image(image, use_column_width=True)
        with col2:
            st.write("Rotated Image")
            st.image(result_img, use_column_width=True)

        st.write("Done!")


if __name__ == "__main__":
    run()
