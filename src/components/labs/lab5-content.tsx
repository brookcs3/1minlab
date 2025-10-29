import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export function Lab5Content() {
  return (
    <>
      <h3>Lab 5: U-Net Architecture - 1-Minute Presentation Guide</h3>
      <h4>How We Present This in Exactly One Minute!</h4>
      <p>
        Let&apos;s master this 1-minute U-Net finale! We&apos;ll show the encoder-decoder architecture, demonstrate complete pipeline integration, and reveal how U-Net automates what was once manual analysis. The culmination of all 5 labs in 60 seconds!
      </p>

      <h4>What the 1-Minute Presentation Will Be</h4>

      <p>
        <strong>Timing: 0:00 - 0:15 (15 seconds): U-Net Building Block</strong>
      </p>
      <p>Code to Run:</p>
      <pre>
        <code>
{`import torch
import torch.nn as nn

# Simple U-Net building block
class SimpleUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# Test the block
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
block = SimpleUNetBlock(in_channels=1, out_channels=16).to(device)

input_tensor = torch.randn(1, 1, 1025, 646).to(device)
output = block(input_tensor)

print("=== U-Net Building Block ===")
print(f"Input: {input_tensor.shape}")
print(f"Output: {output.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")`}
        </code>
      </pre>
      <p>
        <strong>Narration:</strong> "Lab 5: U-Net&apos;s encoder-decoder architecture learns to separate spectrograms. This block processes frequency patterns!"
      </p>
      <p>Expected Output:</p>
      <pre>
        <code>
{`=== U-Net Building Block ===
Input: torch.Size([1, 1, 1025, 646])
Output: torch.Size([1, 16, 1025, 646])
Parameters: 448`}
        </code>
      </pre>

      <p>
        <strong>Timing: 0:15 - 0:45 (30 seconds): Complete Pipeline Integration</strong>
      </p>
      <p>Code to Run:</p>
      <pre>
        <code>
{`import librosa
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Processing on: {device}\\n")

# ==========================================
# COMPLETE PIPELINE: Labs 1-5 Integration
# ==========================================

print("=== Step 1-3: Audio → Spectrogram ===")
audio, sr = librosa.load('sample_audio.wav', sr=22050)
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)
phase = np.angle(stft)
print(f"Audio: {audio.shape} → Spectrogram: {magnitude.shape}")

print("\\n=== Step 4: Convert to PyTorch ===")
tensor = torch.from_numpy(magnitude).float()
tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
print(f"Tensor: {tensor.shape}, Device: {tensor.device}")

print("\\n=== Step 5: U-Net Processing ===")
# Simulate U-Net: create vocal and music masks
vocal_mask = torch.sigmoid(torch.randn_like(tensor))  # 0-1 mask
music_mask = 1 - vocal_mask

print(f"Vocal mask: {vocal_mask.shape}")
print(f"Music mask: {music_mask.shape}")

print("\\n=== Step 6-7: Apply Masks & Reconstruct ===")
vocal_mask_np = vocal_mask.squeeze().cpu().numpy()
music_mask_np = music_mask.squeeze().cpu().numpy()

magnitude_vocals = magnitude * vocal_mask_np
magnitude_music = magnitude * music_mask_np

stft_vocals = magnitude_vocals * np.exp(1j * phase)
stft_music = magnitude_music * np.exp(1j * phase)

vocals = librosa.istft(stft_vocals, hop_length=512, length=len(audio))
music = librosa.istft(stft_music, hop_length=512, length=len(audio))

print(f"Separated vocals: {vocals.shape}")
print(f"Separated music: {music.shape}")`}
        </code>
      </pre>
      <p>
        <strong>Narration:</strong> "All 5 labs integrated! Audio → spectrogram → tensor → U-Net masks → separated vocals and music. U-Net automates the POC&apos;s manual analysis!"
      </p>
      <p>Expected Output:</p>
      <pre>
        <code>
{`Processing on: cuda

=== Step 1-3: Audio → Spectrogram ===
Audio: (661500,) → Spectrogram: (1025, 646)

=== Step 4: Convert to PyTorch ===
Tensor: torch.Size([1, 1, 1025, 646]), Device: cuda:0

=== Step 5: U-Net Processing ===
Vocal mask: torch.Size([1, 1, 1025, 646])
Music mask: torch.Size([1, 1, 1025, 646])

=== Step 6-7: Apply Masks & Reconstruct ===
Separated vocals: (661500,)
Separated music: (661500,)`}
        </code>
      </pre>

      <p>
        <strong>Timing: 0:45 - 1:00 (15 seconds): Wrap-up</strong>
      </p>
      <p>
        <strong>Narration:</strong> "U-Net processes spectrograms to create separation masks. This automates the POC - neural networks learn what was done manually!"
      </p>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="item-1" className="border-t mt-4 pt-4">
          <AccordionTrigger className="text-accent-foreground hover:no-underline text-base font-semibold">View Setup Instructions</AccordionTrigger>
          <AccordionContent>
            <div className="prose max-w-none pt-4">
              <h4>Setup Beforehand (35-40 minutes total prep time)</h4>
              <p>Each team member will assemble their own code and working repository for this lab.</p>
              
              <p><strong>Environment Setup (12-15 minutes)</strong></p>
              <ul>
                <li>PyTorch Installation: Install PyTorch with full CUDA support for U-Net operations</li>
                <li>Complete Dependencies: Install librosa, numpy, matplotlib, soundfile (for audio I/O)</li>
                <li>GPU Verification: Test CUDA installation and memory availability</li>
                <li>U-Net Dependencies: Ensure all neural network components are available</li>
              </ul>

              <p><strong>Code Assembly and Repository Setup (18-20 minutes)</strong></p>
              <ul>
                <li>Create Working Directory: Set up comprehensive Lab 5 repository structure</li>
                <li>Implement U-Net Components:
                  <ul>
                    <li>Create <code>lab5_unet_demo.py</code> with classes: <code>SimpleUNetBlock</code>, functions: <code>complete_separation_pipeline()</code></li>
                    <li>Include encoder-decoder architecture simulation</li>
                    <li>Add complete pipeline integration (Labs 1-5)</li>
                    <li>Implement mask generation and audio reconstruction</li>
                  </ul>
                </li>
                <li>Independent Testing: Test U-Net blocks and complete pipeline</li>
                <li>Integration Verification: Ensure all lab components work together</li>
                <li>Performance Testing: Benchmark pipeline execution time</li>
                <li>Documentation: Add comprehensive docstrings and comments</li>
                <li>Version Control: Commit complete working implementation</li>
              </ul>

              <p><strong>Data and Model Assets Preparation (8-10 minutes)</strong></p>
              <ul>
                <li>Audio File Selection: Choose representative mixed audio for separation demo</li>
                <li>Pipeline Testing: Pre-run complete pipeline with test data</li>
                <li>Output Validation: Test separated audio stems for quality</li>
                <li>Simulation Setup: Prepare U-Net output simulation (since training not required for demo)</li>
              </ul>

              <p><strong>Hardware and Performance Verification (5-7 minutes)</strong></p>
              <ul>
                <li>GPU Requirements: Verify sufficient VRAM for U-Net tensor operations</li>
                <li>Pipeline Performance: Time complete audio separation pipeline</li>
                <li>Memory Management: Test large tensor handling and cleanup</li>
                <li>Fallback Options: Ensure CPU-only execution works (slower but functional)</li>
              </ul>

              <p><strong>Presentation Materials Organization (1-2 minutes)</strong></p>
              <ul>
                <li>Slide Preparation: Access Slide 3 (U-Net blocks) and Slide 4 (integrated pipeline)</li>
                <li>Demo Timing: Practice complete 1-minute presentation</li>
                <li>Backup Content: Prepare training vs inference explanations</li>
              </ul>
            </div>
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="item-2" className="border-t mt-4 pt-4">
          <AccordionTrigger className="text-accent-foreground hover:no-underline text-base font-semibold">View Additional Details</AccordionTrigger>
          <AccordionContent>
            <div className="prose max-w-none pt-4">
              <h4>Key Points to Emphasize</h4>
              <ul>
                <li>U-Net encoder-decoder architecture for spectrogram processing</li>
                <li>Complete integration of all 5 labs</li>
                <li>Automation of the POC&apos;s manual 18-slice analysis</li>
                <li>Pipeline: Audio → Spectrogram → Tensor → Masks → Separated Audio</li>
              </ul>

              <h4>Troubleshooting</h4>
              <ul>
                <li>If GPU unavailable: Use CPU (slower but works)</li>
                <li>If memory issues: Reduce tensor sizes</li>
                <li>If audio processing slow: Use shorter audio file</li>
              </ul>

              <h4>Success Criteria</h4>
              <ul>
                <li>[ ] U-Net block demonstrates convolution operations</li>
                <li>[ ] Complete pipeline shows all 5 labs integrated</li>
                <li>[ ] Separation masks created and applied</li>
                <li>[ ] Audio reconstruction completed</li>
                <li>[ ] Concepts explained within 1 minute</li>
              </ul>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </>
  );
}
