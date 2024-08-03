import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import io

# Function to apply filters to images
def apply_filter(img, filter_type):
    if filter_type == 'Grayscale':
        return ImageOps.grayscale(img)
    elif filter_type == 'Enhance Contrast':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(2)
    return img

# Streamlit application
st.title('Interactive Media Application ')
st.image("images (1).jpg")


# Sidebar
st.sidebar.header('Settings')
filter_option = st.sidebar.selectbox("Apply Filter:", ["None", "Grayscale", "Enhance Contrast"])
st.sidebar.text('Upload a media file or use other features.')

# Sidebar options
sidebar_option = st.sidebar.selectbox("Select a feature:", ["Upload Media", "Questionnaire", "Graphs", "Data Visualization"])

if sidebar_option == "Upload Media":
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg", "mp4", "wav"])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            image = Image.open(uploaded_file)
            if filter_option != 'None':
                image = apply_filter(image, filter_option)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
        elif file_type == 'video':
            st.video(uploaded_file)
            
        elif file_type == 'audio':
            st.audio(uploaded_file)
        
elif sidebar_option == "Questionnaire":
    st.header("User Questionnaire")
    st.write("Please answer the following questions:")
    
    # Form for user inputs
    with st.form(key='questionnaire_form'):
        name = st.text_input("Name")
        age = st.slider("Age", min_value=0, max_value=100, value=25)
        gender = st.radio("Gender", ["Male", "Female", "Other"])
        feedback = st.text_area("Feedback")
        
        # Submit button
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            st.write(f"Name: {name}")
            st.write(f"Age: {age}")
            st.write(f"Gender: {gender}")
            st.write(f"Feedback: {feedback}")

elif sidebar_option == "Graphs":
    st.header("Graphs and Data Visualization")
    
    # Example data for graph
    data = {
        'Category A': np.random.normal(loc=50, scale=10, size=100),
        'Category B': np.random.normal(loc=30, scale=5, size=100),
        'Category C': np.random.normal(loc=70, scale=15, size=100)
    }
    
    # Display histogram
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    for category, values in data.items():
        ax.hist(values, bins=20, alpha=0.5, label=category)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Categories')
    ax.legend()
    st.pyplot(fig)



    
    # Display line chart with user input
    st.subheader("Line Chart")
    x = st.slider('Select number of points', min_value=10, max_value=100, value=50)
    y = np.random.normal(loc=0, scale=1, size=x)
    st.line_chart(pd.DataFrame({'X': np.arange(x), 'Y': y}))

elif sidebar_option == "Data Visualization":
    st.header("Interactive Data Visualization")

    # Dynamic data input
    st.subheader("Enter data for plotting")
    data_input = st.text_area("Paste data (comma-separated values):", "10,20,30,40,50")
    try:
        data = np.array([float(i) for i in data_input.split(',')])
    except ValueError:
        st.error("Please enter valid numeric data.")
        data = np.array([])

    if data.size > 0:
        st.write("Data Entered:")
        st.write(data)
        
        # Plot data
        st.subheader("Bar Chart")
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(data)), data, color='skyblue')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Bar Chart of Entered Data')
        st.pyplot(fig)

else:
    st.write("Please select an option from the sidebar.")

    df= pd.DataFrame(    
        np.random.randn(10, 2),   
          columns=['x', 'y'])
    st.area_chart(df)
