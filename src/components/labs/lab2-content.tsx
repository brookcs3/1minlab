export function Lab2Content() {
  return (
    <>
      <h3>NumPy for Audio Slicing</h3>
      <p>
        NumPy is a powerful library for numerical operations. In audio processing, we often represent sound as a NumPy array. This lab demonstrates how to slice an audio signal into multiple segments for analysis.
      </p>
      <h3>Volume Operations</h3>
      <p>
        We can easily adjust the volume of an audio signal by multiplying the NumPy array by a scalar value. This is a fundamental operation in audio engineering.
      </p>
      <pre>
        <code>
{`import numpy as np

def change_volume(signal: np.ndarray, factor: float) -> np.ndarray:
    """Changes the volume of a signal."""
    return signal * factor

# Example: Halve the volume
audio_signal = np.random.randn(44100) # 1 second of audio at 44.1kHz
quieter_signal = change_volume(audio_signal, 0.5)

print(f"Original signal mean power: {np.mean(audio_signal**2):.4f}")
print(f"Quieter signal mean power: {np.mean(quieter_signal**2):.4f}")`}
        </code>
      </pre>
      <h3>18-Slice Analysis</h3>
      <p>
        For detailed analysis, we might divide our audio into a specific number of slices. Here's how you can divide an audio signal into 18 equal parts using NumPy.
      </p>
      <pre>
        <code>
{`def slice_audio(signal: np.ndarray, num_slices: int = 18) -> list:
    """Slices an audio signal into N equal parts."""
    return np.array_split(signal, num_slices)

slices = slice_audio(audio_signal, 18)
print(f"\\nCreated {len(slices)} slices.")
print(f"Shape of first slice: {slices[0].shape}")`}
        </code>
      </pre>
    </>
  );
}
