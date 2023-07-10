# Load the model
model = keras.models.load_model("Zirak")

# Preprocess input file
audio_file = "Add your Audio file directory..."
data, sr = librosa.load(audio_file)

# Extract features from the input file
stft = librosa.stft(data)
stft_mag = np.abs(stft)
spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
chroma = librosa.feature.chroma_stft(y=data, sr=sr)
zero_cross_rate = librosa.zero_crossings(data, pad=False)

# Resize the feature arrays to have the same number of samples
min_length = min(len(stft_mag), len(spectral_rolloff), len(chroma.T), len(zero_cross_rate))
stft_mag = stft_mag[:, :min_length]
spectral_rolloff = spectral_rolloff[:min_length]
chroma = chroma.T[:min_length]
zero_cross_rate = zero_cross_rate[:min_length]

# Concatenate the extracted features
input_features = np.hstack((stft_mag.T, spectral_rolloff.reshape(-1, 1), chroma, zero_cross_rate.reshape(-1, 1)))

# Standardize the input features using the same scaler used in training
input_features = fit.transform(input_features[:, :58])  # Keep only the first 58 features

# Make predictions
predictions = model.predict(input_features)
predicted_genre_index = np.argmax(predictions[0])
predicted_genre = convertor.inverse_transform([predicted_genre_index])[0]
print("Predicted Genre:", predicted_genre)

# Plot the predicted genre probabilities
genres = ['blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Get the predicted genre probabilities
prediction_probs = predictions[0]

# Plot the predicted genre probabilities
plt.figure(figsize=(10, 6))
plt.bar(genres, prediction_probs)
plt.xlabel("Genre")
plt.ylabel("Probability")
plt.title("Predicted Genre Probabilities")
plt.show()
