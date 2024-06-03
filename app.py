import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cohere
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Post Analysis and Generation", layout="centered")

# Load the environment variables
load_dotenv()

# Cohere API key
cohere_api_key = "UqGv7K0T4Kyhhcy7bzBq1lNStTz9C8moo8aQIvsY"
co = cohere.Client(cohere_api_key)

def caption_generator(des, num_captions=1):
    caption_prompt = f'''Task: You are an expert caption writer tasked with creating engaging and descriptive captions for image descriptions. Your goal is to craft captions that capture the essence of the image, evoke emotions, and provide context or a narrative that enhances the viewer's experience.

image descriptions: {des}
number of caption: {num_captions}
Requirements:
- Analyze the image description properly, look for a group of students, classrooms, computer labs, fun moments, pets, etc.
- Consider the potential emotions, mood, or story the image conveys
- Craft a caption that is concise yet descriptive, capturing the essence of the image
- Incorporate relevant hashtags or keywords that relate to the image's content or theme
- Aim for a caption length of 1-3 sentences (around 20-50 words)

Guidelines:
- Use an engaging tone and descriptive language to bring the image to life
- Avoid stating the obvious or simply listing the objects in the image
- Provide context, a narrative, or a thought-provoking perspective
- Consider the target audience and platform (e.g., Instagram, Twitter, blog post)
- Maintain a consistent voice and style throughout the caption

Please generate a compelling and descriptive caption for the provided image.'''
    response = co.generate(
        model='command-r-plus',
        prompt=caption_prompt,
        max_tokens=200,
        temperature=0.7,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    captions = response.generations[0].text.strip().split('\n')
    return captions

def hashtag_generator(des):
    hashtag_prompt = f'''Task: You are an expert hashtag generator tasked with identifying relevant hashtags for images based on their descriptions. Your goal is to suggest a set of hashtags that accurately capture the content, theme, and context of the image, making it easier for users to discover and engage with the content.

Image Description: {des}

Requirements:
- Carefully analyze the provided image description to understand the subject, setting, and notable elements
- Identify key topics, themes, or concepts present in the description
- Generate a diverse set of relevant hashtags that accurately represent the image's content
- Include both broad and specific hashtags to cater to different search patterns
- Consider popular and trending hashtags related to the image's subject matter

Guidelines:
- Aim to generate between 5-10 hashtags for the given image description
- Use a combination of single-word and multi-word hashtags for better discoverability
- Avoid using overly generic or irrelevant hashtags
- Ensure hashtags are spelled correctly and follow appropriate formatting (e.g., #NoSpaces, #CamelCase)
- Consider the target audience and platform when selecting hashtags
- make sure to directly mention the hashtags and not even mention a single word besides hashtags.

Please provide a list of relevant and accurate hashtags based on the provided image description.
'''
    response = co.generate(
        model='command-r-plus',
        prompt=hashtag_prompt,
        max_tokens=100,
        temperature=0.7,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    hashtags = response.generations[0].text.strip().split(' ')
    return hashtags

def get_image_description(image):
    co = cohere.Client(cohere_api_key)
    response = co.generate(
        model='command-r-plus',
        prompt=f'''Task: You are an expert image analysis AI assistant. Your goal is to provide a highly detailed and accurate description of an image, covering all relevant objects, scenes, colors, textures, and other visual elements present. Your description should be dense and thorough, leaving no significant aspect of the image unmentioned.

Image: {image}

Requirements:
- Describe all major objects, people, scenes, and visual elements in the image, look for a group of students, classrooms, computer labs, fun moments, pets, etc
- Mention notable colors, textures, patterns, and materials present
- Provide details on lighting, perspective, composition, and depth
- Identify any text, logos, or symbols if applicable
- Aim for a comprehensive yet concise description capturing the essence of the image
- Your response should be between 100-170 words in length
- make sure to directly mention the caption and not even mention a single word besides the caption.


Please provide your highly detailed and accurate image description.
''',
        max_tokens=100,
        temperature=0.7,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
    description = response.generations[0].text.strip()
    return description

def kmeans_clustering(data):
    # Select relevant features
    features = ["Impressions", "Saves", "Comments", "Shares", "Likes", "Profile Visits", "Follows"]
    X = data[features]

    # Handle missing values or outliers if needed

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Plot to determine the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Determine the number of clusters (k) based on the elbow plot
    k = 3  # Example: 3 clusters

    # Fit the K-Means algorithm
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # Cluster Feature Comparison
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=data, x=feature, hue=labels, shade=True, common_norm=False)
        plt.title(f'Distribution of {feature} by Cluster', fontsize=16)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        # st.pyplot(plt)

    # Analyze cluster centers and assign descriptive cluster labels
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    overall_mean = X.mean()
    cluster_labels = []
    for i in range(k):
        center = cluster_centers[i]
        higher_count = sum(center > overall_mean)
        if higher_count >= len(features) * 2 / 3:
            cluster_labels.append('High Performing Posts')
        elif higher_count <= len(features) / 3:
            cluster_labels.append('Low Performing Posts')
        else:
            cluster_labels.append('Average Performing Posts')

    # Function to classify new data with descriptive labels
    def classify_new_data(impressions, saves, comments, shares, likes, profile_visits, follows):
        new_data = np.array([[impressions, saves, comments, shares, likes, profile_visits, follows]])
        new_data_scaled = scaler.transform(new_data)
        cluster_label = kmeans.predict(new_data_scaled)[0]
        cluster_name = cluster_labels[cluster_label]
        return cluster_name

    return classify_new_data

def main():
    # Set page config
    # st.set_page_config(page_title="Post Analysis and Generation", layout="centered")

    # Add custom CSS styles
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
            font-family: 'Montserrat', sans-serif;
            margin: 0;
            padding: 0;
        }
        h1 {
            color: #333333;
            font-size: 36px;
            text-align: center;
            margin-top: 30px;
        }
        .caption {
            color: #ffffff;
            font-size: 18px;
            background-color: #333333;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .hashtag {
            color: #ffffff;
            font-size: 14px;
            background-color: #333333;
            padding: 8px;
            border-radius: 4px;
            margin-right: 8px;
            display: inline-block;
        }
        .footer {
            color: #666666;
            font-size: 12px;
            text-align: center;
            margin-top: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.title("Project II - Post Analysis and Generation")

    # Option selection
    option = st.selectbox("Select an option", ["For Already Posted Posts", "For New Posts"])

    # Initialize data as None
    data = None


    if option == "For Already Posted Posts":
    # Load your data here
        data = pd.read_csv("C:/Users/pgupt/Desktop/project 2/Instagram.csv", encoding = 'latin1')
        classify_new_data = kmeans_clustering(data)

        st.subheader("Classify New Post")
        new_impressions = st.number_input("New Impressions", value=10000, step=1)
        new_saves = st.number_input("New Saves", value=20, step=1)
        new_comments = st.number_input("New Comments", value=50, step=1)
        new_shares = st.number_input("New Shares", value=5, step=1)
        new_likes = st.number_input("New Likes", value=800, step=1)
        new_profile_visits = st.number_input("New Profile Visits", value=40, step=1)
        new_follows = st.number_input("New Follows", value=10, step=1)

        if st.button("Classify Post"):
            cluster_result = classify_new_data(new_impressions, new_saves, new_comments, new_shares, new_likes, new_profile_visits, new_follows)
            st.markdown(f"<h2 style='font-weight:bold;'>The new data point belongs to: {cluster_result}</h2>", unsafe_allow_html=True)


            st.subheader("Data on which this model is trained")
            st.write(data)  # Display the data DataFrame

            # Cluster Feature Comparison
            features = ["Impressions", "Saves", "Comments", "Shares", "Likes", "Profile Visits", "Follows"]
            X = data[features]

            # Handle missing values or outliers if needed

            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit the K-Means algorithm
            k = 3  # Example: 3 clusters
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            labels = kmeans.labels_

            # Cluster Feature Comparison
            for feature in features:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=data, x=feature, hue=labels, shade=True, common_norm=False)
                plt.title(f'Distribution of {feature} by Cluster', fontsize=16)
                plt.xlabel(feature, fontsize=14)
                plt.ylabel('Density', fontsize=14)
                st.pyplot(plt)



    elif option == "For New Posts":
        st.subheader("Get Captions and Hashtags for your Image")

        # Image uploader
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            # Display the uploaded image
            img_pil = Image.open(uploaded_image)
            st.image(img_pil, width=400)

            # Generate description, captions, and hashtags
            description = get_image_description(img_pil)
            captions = caption_generator(description)
            hashtags = hashtag_generator(description)

            # Display captions
            st.subheader("Captions for this image:")
            for caption in captions:
                st.markdown(f"<div class='caption'>{caption}</div>", unsafe_allow_html=True)

            # Display hashtags
            st.subheader("#Hashtags")
            hashtag_container = st.container()
            for hash_tag in hashtags:
                hashtag_container.markdown(f"<div class='hashtag'>{hash_tag}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>By 0101</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()