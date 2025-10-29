import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Code, MessageSquare, Terminal, Info, CheckCircle, AlertTriangle, Wrench } from "lucide-react";

export function Lab3Content() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold font-headline">
            Lab 3: librosa STFT and Spectrograms
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4">
            Let's nail this 1-minute STFT demonstration! We'll transform audio waveforms into spectrograms with live coding, show the perfect round-trip reconstruction, and reveal how this creates the 2D data that U-Net processes. Technical magic in 60 seconds!
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold font-headline">Timing: 0:00 - 0:15 (15 seconds): Introduction & STFT Transformation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
            <pre><code>{`import librosa
import numpy as np

# Load audio
audio, sr = librosa.load('sample_audio.wav', sr=22050)
print(f"Audio loaded: {audio.shape} samples")

# STFT transformation
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)
phase = np.angle(stft)

print(f"STFT shape: {stft.shape} (complex)")
print(f"Magnitude shape: {magnitude.shape}")
print(f"Phase shape: {phase.shape}")`}</code></pre>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
            <p className="pl-6 border-l-2 border-accent ml-2">"Lab 3: STFT transforms 1D audio into 2D spectrograms! This is the core of the POC and our U-Net pipeline."</p>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Terminal size={16} /> Expected Output:</h4>
            <pre><code>{`Audio loaded: (661500,) samples
STFT shape: (1025, 1293) (complex)
Magnitude shape: (1025, 1293)
Phase shape: (1025, 1293)`}</code></pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold font-headline">Timing: 0:15 - 0:35 (20 seconds): Visualize Spectrogram</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
            <pre><code>{`import librosa.display
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Magnitude spectrogram
librosa.display.specshow(
    librosa.amplitude_to_db(magnitude, ref=np.max),
    sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=axes[0]
)
axes[0].set_title('Magnitude Spectrogram (dB)')
fig.colorbar(axes[0].images[0], ax=axes[0], format='%+2.0f dB')

# Phase spectrogram
librosa.display.specshow(
    phase, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=axes[1], cmap='twilight'
)
axes[1].set_title('Phase Spectrogram')
fig.colorbar(axes[1].images[0], ax=axes[1])

plt.tight_layout()
plt.show()`}</code></pre>
          </div>
          <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
              <p className="pl-6 border-l-2 border-accent ml-2">"Look at this! Magnitude shows loudness, phase shows timing. Bright areas are vocal frequencies - this is image-like data for U-Net."</p>
          </div>
           <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><Info size={16} /> Key Observation:</h4>
              <p className="pl-6 border-l-2 border-accent ml-2">Point out frequency patterns and color differences between magnitude/phase.</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold font-headline">Timing: 0:35 - 0:55 (20 seconds): Round-Trip Reconstruction</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
            <pre><code>{`# Reconstruct audio with ISTFT
reconstructed = librosa.istft(stft, hop_length=512, length=len(audio))

# Check quality
difference = np.abs(audio - reconstructed).max()
mean_diff = np.abs(audio - reconstructed).mean()

print(f"Original length: {len(audio)}")
print(f"Reconstructed length: {len(reconstructed)}")
print(f"Max difference: {difference:.2e}")
print(f"Mean difference: {mean_diff:.2e}")
print(f"Quality: {'Perfect!' if difference < 1e-5 else 'Very Good'}")`}</code></pre>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
            <p className="pl-6 border-l-2 border-accent ml-2">"ISTFT reverses STFT perfectly! This lossless transformation enables our complete audio separation pipeline."</p>
          </div>
          <div>
            <h4 className="font-semibold flex items-center gap-2 mb-2"><Terminal size={16} /> Expected Output:</h4>
            <pre><code>{`Original length: 661500
Reconstructed length: 661500
Max difference: 1.23e-13
Mean difference: 1.45e-14
Quality: Perfect!`}</code></pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold font-headline">Timing: 0:55 - 1:00 (5 seconds): Wrap-up</CardTitle>
        </CardHeader>
        <CardContent>
          <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
          <p className="pl-6 border-l-2 border-accent ml-2">"STFT/ISTFT is reversible - the foundation for U-Net audio processing!"</p>
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
              <h4>Setup Beforehand (25-30 minutes total prep time)</h4>
              <p>Each team member will assemble their own code and working repository for this lab.</p>

              <p><strong>Environment Setup (7-10 minutes)</strong></p>
              <ul>
                <li>Install Dependencies: <code>pip install librosa numpy matplotlib</code></li>
                <li>librosa Verification: Test STFT/ISTFT operations with sample data</li>
                <li>Visualization Setup: Configure matplotlib for spectrogram display with proper colorbars</li>
              </ul>

              <p><strong>Code Assembly and Repository Setup (12-15 minutes)</strong></p>
              <ul>
                <li>Create Working Directory: Set up Lab 3 repository with git initialization</li>
                <li>Implement STFT Functions:
                  <ul>
                    <li>Create <code>lab3_stft_basics.py</code> with functions: <code>create_spectrogram()</code>, <code>visualize_spectrogram()</code>, <code>test_round_trip()</code></li>
                    <li>Include magnitude/phase separation and reconstruction logic</li>
                    <li>Add parameter exploration (nfft, hoplength variations)</li>
                  </ul>
                </li>
                <li>Independent Testing: Test round-trip reconstruction accuracy</li>
                <li>Performance Optimization: Experiment with different STFT parameters</li>
                <li>Version Control: Commit tested code with meaningful commit messages</li>
              </ul>

              <p><strong>Audio Assets and Test Data Preparation (5-7 minutes)</strong></p>
              <ul>
                <li>Audio File Selection: Choose 30-second audio file for consistent demos</li>
                <li>Multiple Test Files: Prepare different audio types (speech, music, mixed)</li>
                <li>Pre-computed Spectrograms: Generate spectrograms for faster demo loading</li>
                <li>Quality Verification: Test reconstruction quality across different audio types</li>
              </ul>

              <p><strong>Hardware and Visualization Verification (4-5 minutes)</strong></p>
              <ul>
                <li>STFT Performance: Test computation time on available hardware</li>
                <li>Memory Requirements: Verify sufficient RAM for spectrogram arrays</li>
                <li>Display Quality: Ensure spectrogram plots render with proper frequency/time axes</li>
                <li>Color Scheme Testing: Verify dB scaling and colorbar functionality</li>
              </ul>

              <p><strong>Presentation Materials Organization (1-2 minutes)</strong></p>
              <ul>
                <li>Slide Preparation: Access Slide 3 (STFT demo) and Slide 4 (ISTFT reconstruction)</li>
                <li>Demo Timing Practice: Run complete sequence to confirm 1-minute execution</li>
                <li>Backup Slides: Prepare STFT parameter explanations</li>
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
                <li>STFT as time-frequency transformation</li>
                <li>Spectrograms as 2D representations for neural networks</li>
                <li>Perfect reconstruction with ISTFT</li>
                <li>Connection to the POC manual analysis</li>
              </ul>

              <h4 className="flex items-center gap-2 mt-4"><AlertTriangle size={16} /> Troubleshooting</h4>
              <ul>
                <li>If STFT slow: Reduce <code>n_fft</code> or use shorter audio</li>
                <li>If visualization issues: Check matplotlib backend</li>
                <li>If reconstruction imperfect: Verify <code>hop_length</code> consistency</li>
              </ul>

              <h4 className="flex items-center gap-2 mt-4"><CheckCircle size={16} /> Success Criteria</h4>
              <ul>
                <li>[ ] STFT computed with correct shapes</li>
                <li>[ ] Spectrogram visualized with magnitude and phase</li>
                <li>[ ] Round-trip reconstruction shows near-zero error</li>
                <li>[ ] Transformation concepts explained clearly</li>
              </ul>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
