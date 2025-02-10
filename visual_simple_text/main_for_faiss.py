from langchain.embeddings import HuggingFaceEmbeddings

import gradio as gr
from langchain_community.vectorstores import FAISS

from sklearn.manifold import TSNE

import plotly.graph_objects as go
import numpy as np


MODEL = "deepseek-chat"

# store text list which to be vectorized
text_list = [
    "Apple", "Macbookpro", "IPhone", "Company", "Steve Jobs", "Microsoft", "微软", "苹果",
    "水果", "市值万亿美元的公司", "史蒂夫乔布斯", "蒂姆库克", "Windows", "Ipod Touch", "Steva Jobs",
    "Mac Mini", "Apple Watch", "iPad", "iOS", "macOS", "App Store", "iCloud", "Siri",
    "Apple Music", "Apple Pay", "Apple Store", "乔纳森艾维", "Apple Park", "Apple Car",
    "Apple Silicon", "M1芯片", "AirPods", "HomePod", "Apple TV", "Face ID", "Touch ID",
    "Retina Display", "Apple Pencil", "Apple Arcade", "Apple Fitness+"
]

# insert text
def insert_text_to_list(new_text):
    if new_text:
        text_list.append(new_text)
    return text_list  # return inserted list

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def vector_text_list_to_2d(to_be_vector_list: []):
    to_be_vector_list = text_list
    vectorstore = FAISS.from_texts(to_be_vector_list, embedding=embeddings)
    total_vectors = vectorstore.index.ntotal
    dimensions = vectorstore.index.d
    print(f"total vectors: {total_vectors}  dimensions: {dimensions}")

    vectors = []
    for i in range(total_vectors):
        vectors.append(vectorstore.index.reconstruct(i))
    vectors = np.array(vectors)

    tsne = TSNE(n_components=2, random_state=42)

    reduced_vectors = tsne.fit_transform(vectors)


    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        text=[f"{text}" for text in to_be_vector_list],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='2D FAISS Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()


def vector_text_list_to_3d(to_be_vector_list: []):
    to_be_vector_list = text_list
    vectorstore = FAISS.from_texts(text_list, embedding=embeddings)
    total_vectors = vectorstore.index.ntotal
    dimensions = vectorstore.index.d
    print(f"total vectors: {total_vectors}  dimensions: {dimensions}")

    vectors = []
    for i in range(total_vectors):
        vectors.append(vectorstore.index.reconstruct(i))
    vectors = np.array(vectors)

    tsne = TSNE(n_components=3, random_state=42)

    reduced_vectors = tsne.fit_transform(vectors)


    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        text=[f"{text}" for text in to_be_vector_list],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='3D FAISS Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Visual Simple Text")

    # Displays a list of submitted text
    with gr.Row():
        text_display = gr.Textbox(label="To Be Vector Text",
                                  value=str(text_list),
                                  lines=10, interactive=False)

    # Input Text
    with gr.Row():
        text_input = gr.Textbox(label="Please enter the text you want to vectorize", placeholder="please input text")

    # Submit button
    with gr.Row():
        submit_btn = gr.Button("Add")

    with gr.Row():
        visual_to_2d_btn = gr.Button("start visual vector to 2d")
        visual_to_3d_btn = gr.Button("start visual vector to 3d")


    # Binding submit btn click
    submit_btn.click(
        fn=insert_text_to_list,
        inputs=text_input,
        outputs=text_display
    )

    visual_to_2d_btn.click(
        fn=vector_text_list_to_2d,
    )

    visual_to_3d_btn.click(
        fn=vector_text_list_to_3d,
    )


demo.launch()