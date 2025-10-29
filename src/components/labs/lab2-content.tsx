import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Code, MessageSquare, Terminal, Info, CheckCircle, AlertTriangle, Wrench } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function Lab2Content() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold font-headline">
            Lab 2: NumPy for Audio Processing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4">
            Let's dominate this 1-minute NumPy demo! We'll show how NumPy makes audio processing lightning-fast with vectorized operations, create audio signals, and demonstrate the 18-slice analysis that powers our measurements. Live coding at its finest!
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold font-headline">Timing: 0:00 - 0:15 (15 seconds): Introduction & Generate Signal</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
            <pre><code>{`import numpy as np
import matplotlib.pyplot as plt

# Create a simple audio signal (440 Hz sine wave)
sample_rate = 22050
duration = 1.0  # seconds
frequency = 440  # Hz (A note)

t = np.linspace(0, duration, int(sample_rate * duration))
audio_signal = np.sin(2 * np.pi * frequency * t)

print(f"Created {len(audio_signal)} samples")
print(f"Signal range: [{audio_signal.min():.3f}, {audio_signal.max():.3f}]")`}</code></pre>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
            <p className="pl-6 border-l-2 border-accent ml-2">"Lab 2 shows NumPy's power! We create 22,050 samples of a 440 Hz sine wave in one operation."</p>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Terminal size={16} /> Expected Output:</h4>
            <pre><code>{`Created 22050 samples
Signal range: [-1.000, 1.000]`}</code></pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold font-headline">Timing: 0:15 - 0:35 (20 seconds): Volume Operations</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
            <pre><code>{`# Volume adjustments
quiet_signal = audio_signal * 0.5  # Reduce volume
loud_signal = audio_signal * 2.0   # Increase volume

print(f"Original range: [{audio_signal.min():.3f}, {audio_signal.max():.3f}]")
print(f"Quiet range: [{quiet_signal.min():.3f}, {quiet_signal.max():.3f}]")
print(f"Loud range: [{loud_signal.min():.3f}, {loud_signal.max():.3f}]")

# Visualize
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(audio_signal[:500])
plt.title("Original Signal")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(quiet_signal[:500])
plt.title("Quiet Signal (×0.5)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(loud_signal[:500])
plt.title("Loud Signal (×2.0)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()`}</code></pre>
          </div>
          <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
              <p className="pl-6 border-l-2 border-accent ml-2">"Volume changes are simple multiplication! NumPy processes the entire array simultaneously - 100x faster than loops."</p>
          </div>
           <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><Info size={16} /> Key Observation:</h4>
              <p className="pl-6 border-l-2 border-accent ml-2">Show the scaled amplitudes and identical shapes.</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold font-headline">Timing: 0:35 - 0:55 (20 seconds): 18-Slice Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
            <pre><code>{`import librosa

# Load audio and create spectrogram
audio, sr = librosa.load('sample_audio.wav', sr=22050)
stft = librosa.stft(audio)
magnitude = np.abs(stft)

print(f"Full spectrogram: {magnitude.shape}")

# 18-slice approach
num_slices = 18
time_frames = magnitude.shape[1]
slice_size = time_frames // num_slices

print(f"Dividing into {num_slices} slices of {slice_size} frames each")

# Show first 3 slices
for i in range(3):
    start = i * slice_size
    end = start + slice_size
    slice_data = magnitude[:, start:end]
    slice_mean = slice_data.mean()
    slice_std = slice_data.std()
    print(f"Slice {i+1}: mean={slice_mean:.4f}, std={slice_std:.4f}")`}</code></pre>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
            <p className="pl-6 border-l-2 border-accent ml-2">"The POC analyzes spectrograms in 18 slices using NumPy array slicing. This enables the 765,000 measurements!"</p>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Terminal size={16} /> Expected Output:</h4>
            <pre><code>{`Full spectrogram: (1025, 646)
Dividing into 18 slices of 35 frames each
Slice 1: mean=0.0234, std=0.0412
Slice 2: mean=0.0198, std=0.0356
Slice 3: mean=0.0251, std=0.0398`}</code></pre>
          </div>
        </CardContent>
      </Card>
      
      <Card>
          <CardHeader>
              <CardTitle className="text-base font-semibold font-headline">Timing: 0:55 - 1:00 (5 seconds): Wrap-up</CardTitle>
          </CardHeader>
          <CardContent>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
              <p className="pl-6 border-l-2 border-accent ml-2">"NumPy makes audio processing fast and efficient - essential for the POC!"</p>
          </CardContent>
      </Card>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="item-1">
          <AccordionTrigger className="text-accent-foreground hover:no-underline">
              <div className="flex items-center gap-2">
                  <Wrench size={16}/> View Setup Instructions
              </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="prose max-w-none pt-4">
            <h4>Setup Beforehand (20-25 minutes total prep time)</h4>
            <p>Each team member will assemble their own code and working repository for this lab.</p>

            <p><strong>Environment Setup (5-7 minutes)</strong></p>
            <ul>
              <li>Install Dependencies: <code>pip install numpy matplotlib librosa</code></li>
              <li>Verify NumPy Installation: Test array operations and vectorization</li>
              <li>IDE Preparation: Set up Python environment with matplotlib backend configured</li>
            </ul>

            <p><strong>Code Assembly and Repository Setup (10-12 minutes)</strong></p>
            <ul>
              <li>Create Working Directory: Establish dedicated Lab 2 folder with git initialization</li>
              <li>Assemble Core Functions:
                <ul>
                  <li>Create <code>lab2_numpy_basics.py</code> with functions: <code>create_sine_wave()</code>, <code>volume_operations()</code>, <code>slice_analysis()</code></li>
                  <li>Implement vectorized operations and array slicing examples</li>
                  <li>Add comprehensive docstrings and error handling</li>
                </ul>
              </li>
              <li>Independent Testing: Test each function with sample inputs</li>
              <li>Performance Benchmarking: Compare NumPy operations vs Python loops</li>
              <li>Commit Working Code: Use git to version control your implementation</li>
            </ul>

            <p><strong>Data and Assets Preparation (4-5 minutes)</strong></p>
            <ul>
              <li>Audio File Preparation: Obtain audio file from Lab 1 or download new sample</li>
              <li>Spectrogram Generation: Pre-compute spectrogram data for 18-slice demo</li>
              <li>Test Data Creation: Generate synthetic audio signals for sine wave demos</li>
            </ul>

            <p><strong>Hardware and Performance Verification (3-4 minutes)</strong></p>
            <ul>
              <li>NumPy Performance: Test array operations on available hardware (CPU/GPU)</li>
              <li>Visualization Setup: Ensure matplotlib plots display correctly with multiple subplots</li>
              <li>Memory Check: Verify sufficient RAM for spectrogram computations</li>
            </ul>

            <p><strong>Presentation Materials Organization (1-2 minutes)</strong></p>
            <ul>
              <li>Slide Access: Prepare Slide 3 (signal generation) and Slide 4 (18-slice analysis)</li>
              <li>Demo Script Practice: Run complete demo sequence once for timing</li>
              <li>Backup Content: Prepare broadcasting examples if needed</li>
            </ul>
            </div>
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="item-2">
          <AccordionTrigger className="text-accent-foreground hover:no-underline">
              <div className="flex items-center gap-2">
                  <Info size={16}/> View Additional Details
              </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="prose max-w-none pt-4">
              <h4 className="flex items-center gap-2"><Info size={16} /> Key Points to Emphasize</h4>
              <ul>
              <li>Vectorized operations speed (10-100x faster)</li>
              <li>Broadcasting for element-wise operations</li>
              <li>Array slicing for segmenting data</li>
              <li>Connection to the POC measurement approach</li>
              </ul>

              <h4 className="flex items-center gap-2 mt-4"><AlertTriangle size={16} /> Troubleshooting</h4>
              <ul>
                <li>If audio file missing: Use sine wave generation as fallback</li>
                <li>If memory issues: Reduce array sizes or use smaller audio</li>
                <li>If slow operations: Ensure NumPy is installed correctly</li>
              </ul>

              <h4 className="flex items-center gap-2 mt-4"><CheckCircle size={16} /> Success Criteria</h4>
              <ul>
                <li>[ ] Sine wave generated with correct properties</li>
                <li>[ ] Volume operations demonstrated with plots</li>
                <li>[ ] 18-slice analysis shows slicing and statistics</li>
                <li>[ ] Concepts explained within 1 minute timeframe</li>
              </ul>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
