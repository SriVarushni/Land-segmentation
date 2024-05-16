import os
import streamlit as st
from PIL import Image
import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
import matplotlib.pyplot as plt
import numpy as np

# Function to append unique image filenames to demo.txt
def append_to_demo_txt(uploaded_files):
    appended_file_names = set()
    demo_txt_path = 'samples/list/demo.txt'

    # Read existing content in demo.txt
    with open(demo_txt_path, 'r') as f:
        content = f.readlines()
        appended_file_names.update([line.strip() for line in content])

    # Append unique filenames from uploaded files
    with open(demo_txt_path, 'a') as f:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name not in appended_file_names:
                f.write(f"{file_name}\n")
                appended_file_names.add(file_name)

# Function to clear existing content in demo.txt
def clear_demo_txt():
    open('samples/list/demo.txt', 'w').close()

# Function to run demo.py
def run_demo():
    subprocess.run(["python", "demo.py"])

# Function to run eval_cd.py and capture its output
def run_eval_cd():
    result = subprocess.run(["python", "eval_cd.py"], capture_output=True, text=True)
    return result.stdout

# Function to display images from the predict folder
def display_change_maps():
    predict_folder = 'samples/predict'
    demo_txt_path = 'samples/list/demo.txt'

    # Read file names from demo.txt
    with open(demo_txt_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]

    # Display change maps for each file name
    for file_name in file_names:
        change_map_path = os.path.join(predict_folder, file_name)
        if os.path.exists(change_map_path):
            st.image(change_map_path, caption=file_name, use_column_width=True)
            pdf_path = os.path.join(predict_folder, f"{file_name}.pdf")
            st.markdown(f"[Download PDF]({pdf_path})")  # Provide a download option below each image
        else:
            st.write(f"Change map not found for {file_name}")

def generate_pdf(file_names, eval_output, graph_image):
    predict_folder = 'samples/predict'
    pdf_path = os.path.join(predict_folder, 'output.pdf')

    # Create a single PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Set to store unique images
    unique_images = set()

    for file_name in file_names:
        # Calculate image dimensions to fit the page with a 20 mm border
        image_path = os.path.join(predict_folder, file_name)
        if os.path.exists(image_path) and image_path not in unique_images:
            img = Image.open(image_path)
            img_width, img_height = img.size
            page_width, page_height = letter
            border_width = 10 * mm
            available_width = page_width - 2 * border_width
            available_height = page_height - 2 * border_width
            aspect_ratio = img_width / img_height

            # Calculate width and height of the image within the available space
            if aspect_ratio > 1:
                width = available_width
                height = available_width / aspect_ratio
                if height > available_height:
                    height = available_height
                    width = height * aspect_ratio
            else:
                height = available_height
                width = available_height * aspect_ratio
                if width > available_width:
                    width = available_width
                    height = width / aspect_ratio

            # Calculate x and y coordinates to center the image
            x_offset = (available_width - width) / 2 + border_width
            y_offset = (available_height - height) / 2 + border_width

            # Draw image with a 20 mm border
            c.drawImage(ImageReader(image_path), x_offset, page_height - y_offset - height, width=width, height=height)
            
            # Add image path to the set of unique images
            unique_images.add(image_path)

            # Add evaluation output below the image if available
            if file_name in eval_output and eval_output[file_name]:
                eval_text = eval_output[file_name]
                c.drawString(x_offset, y_offset - 20, eval_text)  # Adjust the position as needed

            c.showPage()  # Add a new page for each image and its output
    
    # Add the graph image to the PDF
    if graph_image:
        c.drawImage(ImageReader(graph_image), 20, 20, width=letter[0] - 40, height=letter[1] - 40)
        c.showPage()  # Add a new page for the graph

    c.save()
    return pdf_path

# Streamlit app
def main():
    st.title('Igress-Change Detection App')

    # File upload widgets
    st.header('Upload Images')
    uploaded_files = st.file_uploader('Upload Before and After Images', type=['png', 'jpg'], accept_multiple_files=True)

    if st.button('Process Images'):
        if uploaded_files:
            # Clear existing content in demo.txt
            clear_demo_txt()

            # Append unique file names to demo.txt
            append_to_demo_txt(uploaded_files)

            # Run demo.py
            run_demo()

            # Run eval_cd.py and capture its output
            eval_result = run_eval_cd()

            # Display generated change maps
            st.header('Generated Change Maps')
            display_change_maps()

            # Generate and provide links to download PDFs
            file_names = [uploaded_file.name for uploaded_file in uploaded_files]
            eval_output = {}
            if eval_result:
                # Parse eval output and store in dictionary
                lines = eval_result.split('\n')
                for line in lines:
                    parts = line.split(':')
                    if len(parts) == 2:
                        eval_output[parts[0].strip()] = parts[1].strip()

            # Generate graph
            plt.figure(figsize=(10, 6))
            # Your graph generation code here
            plt.savefig('samples/predict/graph.png')  # Save the graph as an image
            
            # Generate PDF with images, evaluation outputs, and graph
            graph_image = 'samples/predict/graph.png'
            pdf_paths = generate_pdf(file_names, eval_output, graph_image)

            # Display download button only if there are generated PDF files
            if pdf_paths:
                st.header('Download Combined PDF')
                combined_pdf_path = os.path.join('samples/predict', 'output.pdf')
                st.markdown(f"[Download PDF]({combined_pdf_path})")

if __name__ == '__main__':
    main()
