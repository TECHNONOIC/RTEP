import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode

# Define function to recommend movies based on emotion and genre
def recommend_movies(input_emotion, df):
    # Filter movies based on input emotion
    filtered_by_emotion = df[df['emotion'] == input_emotion.lower()]
    
    if filtered_by_emotion.empty:
        print("No movies found for the input emotion.")
        return []

    # Filter further based on genre
    genres_for_emotion = emotion_genres.get(input_emotion.lower(), [])
    filtered_by_genre = filtered_by_emotion[filtered_by_emotion['genre'].apply(lambda x: any(genre in x for genre in genres_for_emotion))]

    if filtered_by_genre.empty:
        print("No movies found for the input emotion and genre combination.")
        return []

    # Calculate score for each movie
    filtered_by_genre['score'] = filtered_by_genre.apply(lambda row: sum(1 for genre in genres_for_emotion if genre in row['genre']) +
                                                                 sum(2 for word in row['cleaned_overview'].split() if word in input_emotion.lower()), axis=1)
    # Sort the DataFrame by score in descending order
    sorted_df = filtered_by_genre.sort_values(by='score', ascending=False)

    # Get top 20 movies
    recommended_movies = sorted_df.head(20).to_dict('records')

    return recommended_movies

# Function to load DataFrame from .h5 file
def load_dataframe_from_h5(file_path):
    return pd.read_hdf(file_path, key='data')

# Main function to run the recommendation system
def main():
    # Load preprocessed DataFrame from .h5 file
    df_subset = load_dataframe_from_h5("preprocessed_movies.h5")

    # Get the mode emotion from the DataFrame
    input_emotion = mode_emotion

    # Validate the input emotion
    valid_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    if input_emotion.lower() not in valid_emotions:
        print("Invalid emotion detected. Using neutral emotion instead.")
        input_emotion = 'neutral'

    print(f"Detected Emotion: {input_emotion.capitalize()}")

    # Update the 'emotion' column in df_subset
    df_subset['emotion'] = input_emotion.lower()

    # Recommend movies based on the input emotion
    recommended_movies = recommend_movies(input_emotion.lower(), df_subset)

    if recommended_movies:
        print("\nRecommended Movies for", input_emotion.capitalize() + ":")
        for i, movie in enumerate(recommended_movies, start=1):
            print(f"{i}. {movie['original_title']} (Genre: {', '.join(movie['genre'])}) - Score: {movie['score']}")
            print(f"Overview: {movie['overview']}\n")
    else:
        print("No movies found for the input emotion and genre combination.")

# Define genres associated with each emotion
emotion_genres = {
    "angry": ["Thriller", "Action", "Horror"],
    "disgust": ["Horror", "Thriller"],
    "fear": ["Horror", "Thriller", "Mystery"],
    "happy": ["Comedy", "Romance", "Animation"],
    "neutral": ["Drama", "Comedy"],
    "sad": ["Drama", "Romance"],
    "surprise": ["Mystery", "Thriller"]
}

# Load the first CSV file
face_emotions = pd.read_csv(r'E:\GAN for Face expression Classification\final deployment\face_emotions.csv').applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Load the second CSV file
voice_emotions = pd.read_csv(r'E:\GAN for Face expression Classification\final deployment\voiceemotions.csv').applymap(lambda x: x.lower() if isinstance(x, str) else x)

merged_df = pd.merge(face_emotions, voice_emotions, on='time', how='outer')

# Set 'time' column as index
merged_df.set_index('time', inplace=True)

# Fill NaN values with a placeholder, limit to one consecutive NaN value
merged_df_filled = merged_df.fillna(method='ffill', limit=1)

# Reset index
merged_df_filled.reset_index(inplace=True)

# Convert 'time' column to datetime format
merged_df_filled['time'] = pd.to_datetime(merged_df_filled['time'])

# Set 'time' column as index
merged_df_filled.set_index('time', inplace=True)

# Fill NaN values with a placeholder
merged_df_filled = merged_df_filled.fillna('not available')

# Define the function to calculate average emotion
def calculate_average_emotion(row):
    face_emotion = row['face_emotion']
    voice_emotion = row['voice emotion']

    # If both face and voice emotions are NaN, return NaN
    if face_emotion == "not available" and voice_emotion == "not available":
        return "not available"

    # If face emotion is NaN or voice emotion is NaN, use the other value
    elif face_emotion == "not available":
        return voice_emotion

    elif voice_emotion == "not available":
        return face_emotion

    # If both face and voice emotions are available, use only face emotion
    else:
        return face_emotion

merged_df_filled.reset_index(inplace=True)
# Apply the function to calculate average emotion for each row
merged_df_filled['average_emotion'] = merged_df_filled.apply(calculate_average_emotion, axis=1)

# Calculate mode of the average_emotion column
mode_emotion = mode(merged_df_filled['average_emotion'])

# Run the recommendation system
main()
