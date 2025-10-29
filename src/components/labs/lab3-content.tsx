export function Lab3Content() {
  return (
    <>
      <h3>Short-Time Fourier Transform (STFT)</h3>
      <p>
        The STFT is a crucial tool in audio analysis, allowing us to see how the frequency content of a signal changes over time. We use the <code>librosa</code> library to compute the STFT efficiently.
      </p>
      <p>
        <em>It's important my exact text is mentioned. These are tutorials and we need to stress 1 min. Please include my entire text in the view write up.</em>
      </p>
      <pre>
        <code>
{`import librosa
import numpy as np

# Load an example audio file
# Note: In a real app, you would load your own file.
audio, sr = librosa.load(librosa.ex('trumpet'), duration=5)

# Compute the STFT
stft_result = librosa.stft(audio)

# Convert to decibels (log-power scale)
D = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
print(f"Spectrogram shape: {D.shape}")
`}
        </code>
      </pre>
      <h3>Visualizing Spectrograms</h3>
      <p>
        The result of an STFT is a spectrogram, which we can visualize. <code>librosa.display</code> offers convenient functions for this. This visualization is what's often fed into neural networks for audio tasks.
      </p>
      <pre>
        <code>
{`# This code is for illustration. To run it, you need matplotlib.
# import matplotlib.pyplot as plt
# import librosa.display

# fig, ax = plt.subplots(figsize=(10, 4))
# img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Spectrogram of a Trumpet')
# plt.show()
print("Visualization code is ready to be used with matplotlib.")
`}
        </code>
      </pre>
      <h3>Round-Trip Reconstruction</h3>
      <p>
        A key feature is the ability to reconstruct the audio signal from the STFT matrix (a process called inverse STFT). This is essential for tasks like audio separation where we modify the spectrogram before converting it back to sound.
      </p>
      <pre>
        <code>
{`# Inverse STFT to reconstruct the audio
reconstructed_audio = librosa.istft(stft_result)
print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
`}
        </code>
      </pre>
    </>
  );
}
