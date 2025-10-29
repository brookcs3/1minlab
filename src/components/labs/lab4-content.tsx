import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export function Lab4Content() {
  return (
    <>
      <h3>Lab 4: PyTorch Essentials - 1-Minute Presentation Guide</h3>
      <h4>How We Present This in Exactly One Minute!</h4>
      <p>
        Let&apos;s conquer this 1-minute PyTorch showcase! We&apos;ll demonstrate GPU acceleration, seamless NumPy-to-tensor conversion, and the complete pipeline that takes audio all the way to GPU-ready tensors for U-Net processing. Pure neural network power!
      </p>

      <h4>What the 1-Minute Presentation Will Be</h4>

      <p>
        <strong>Timing: 0:00 - 0:15 (15 seconds): PyTorch Setup Check</strong>
      </p>
      <p>Code to Run:</p>
      <pre>
        <code>
{`import torch
import numpy as np

print("=== PyTorch Setup Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\\nSelected device: {device}")`}
        </code>
      </pre>
      <p>
        <strong>Narration:</strong> "Lab 4: PyTorch enables GPU acceleration! This provides the speed needed for U-Net neural networks."
      </p>
      <p>Expected Output (with GPU):</p>
      <pre>
        <code>
{`=== PyTorch Setup Check ===
PyTorch version: 2.0.0
CUDA available: True
CUDA version: 12.1
GPU device: NVIDIA GeForce RTX 2070 Super Max-Q

Selected device: cuda`}
        </code>
      </pre>

      <p>
        <strong>Timing: 0:15 - 0:35 (20 seconds): NumPy to PyTorch Conversion</strong>
      </p>
      <p>Code to Run:</p>
      <pre>
        <code>
{`# Simulate spectrogram data (1025 freq × 646 time)
spectrogram_np = np.random.randn(1025, 646).astype(np.float32)
print(f"NumPy array: shape={spectrogram_np.shape}, dtype={spectrogram_np.dtype}")

# Convert to PyTorch tensor
tensor_cpu = torch.from_numpy(spectrogram_np)
print(f"PyTorch tensor (CPU): {tensor_cpu.shape}, {tensor_cpu.dtype}")

# Move to GPU
tensor_gpu = tensor_cpu.to(device)
print(f"PyTorch tensor (GPU): {tensor_gpu.shape}, device={tensor_gpu.device}")

# Basic neural operation
processed = torch.relu(tensor_gpu) * 0.5
print(f"After processing: {processed.shape}, device={processed.device}")

# Back to NumPy
result_np = processed.cpu().numpy()
print(f"Back to NumPy: {result_np.shape}")`}
        </code>
      </pre>
      <p>
        <strong>Narration:</strong> "Seamless conversion: NumPy spectrograms become PyTorch tensors on GPU. Neural operations happen instantly!"
      </p>
      <p>Expected Output:</p>
      <pre>
        <code>
{`NumPy array: shape=(1025, 646), dtype=float32
PyTorch tensor (CPU): torch.Size([1025, 646]), torch.float32
PyTorch tensor (GPU): torch.Size([1025, 646]), device=cuda:0
After processing: torch.Size([1025, 646]), device=cuda:0
Back to NumPy: (1025, 646)`}
        </code>
      </pre>

      <p>
        <strong>Timing: 0:35 - 0:55 (20 seconds): Complete Pipeline Integration</strong>
      </p>
      <p>Code to Run:</p>
      <pre>
        <code>
{`import librosa

# Step 1-3: Audio → Spectrogram (Labs 1-3)
audio, sr = librosa.load('sample_audio.wav', sr=22050)
stft = librosa.stft(audio, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)

print("=== Complete Pipeline: Audio → GPU Tensor ===")
print(f"1. Audio loaded: {audio.shape}")
print(f"2. Spectrogram: {magnitude.shape}")

# Step 4: Convert to PyTorch (Lab 4)
tensor = torch.from_numpy(magnitude).float()
tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
tensor = tensor.to(device)

print(f"3. PyTorch tensor: {tensor.shape}")
print(f"   Batch: {tensor.shape[0]}, Channels: {tensor.shape[1]}")
print(f"   Frequency: {tensor.shape[2]}, Time: {tensor.shape[3]}")
print(f"   Device: {tensor.device}")
print("\\nReady for U-Net processing!")`}
        </code>
      </pre>
      <p>
        <strong>Narration:</strong> "Complete integration: Audio file → NumPy spectrogram → PyTorch tensor on GPU. This is what U-Net processes!"
      </p>
      <p>Expected Output:</p>
      <pre>
        <code>
{`=== Complete Pipeline: Audio → GPU Tensor ===
1. Audio loaded: (661500,)
2. Spectrogram: (1025, 646)
3. PyTorch tensor: torch.Size([1, 1, 1025, 646])
   Batch: 1, Channels: 1
   Frequency: 1025, Time: 646
   Device: cuda:0

Ready for U-Net processing!`}
        </code>
      </pre>

      <p>
        <strong>Timing: 0:55 - 1:00 (5 seconds): Wrap-up</strong>
      </p>
      <p>
        <strong>Narration:</strong> "PyTorch bridges audio processing to neural networks - GPU acceleration makes U-Net fast!"
      </p>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="item-1" className="border-t mt-4 pt-4">
          <AccordionTrigger className="text-accent-foreground hover:no-underline text-base font-semibold">View Setup Instructions</AccordionTrigger>
          <AccordionContent>
            <div className="prose max-w-none pt-4">
              <h4>Setup Beforehand (30-35 minutes total prep time)</h4>
              <p>Each team member will assemble their own code and working repository for this lab.</p>
              
              <p><strong>Environment Setup (10-12 minutes)</strong></p>
              <ul>
                <li>PyTorch Installation: Install PyTorch with CUDA support if GPU available<br/><code>pip install torch torchvision</code> (or with CUDA: <code>pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118</code>)</li>
                <li>GPU Verification: Test CUDA availability and GPU memory</li>
                <li>Additional Dependencies: Install numpy, librosa for complete pipeline</li>
                <li>Environment Testing: Verify tensor operations and GPU acceleration</li>
              </ul>

              <p><strong>Code Assembly and Repository Setup (15-18 minutes)</strong></p>
              <ul>
                <li>Create Working Directory: Initialize Lab 4 repository with proper structure</li>
                <li>Implement PyTorch Functions:
                  <ul>
                    <li>Create <code>lab4_pytorch_basics.py</code> with functions: <code>check_pytorch_setup()</code>, <code>numpy_to_pytorch_demo()</code>, <code>audio_to_tensor_pipeline()</code></li>
                    <li>Include GPU device selection and tensor shape management</li>
                    <li>Add error handling for CPU-only systems</li>
                  </ul>
                </li>
                <li>Independent Testing: Test NumPy ↔ PyTorch conversions and GPU operations</li>
                <li>Performance Benchmarking: Compare CPU vs GPU tensor operations</li>
                <li>Integration Testing: Verify complete audio → tensor pipeline</li>
                <li>Git Version Control: Commit working code with documentation</li>
              </ul>

              <p><strong>Data Pipeline and Assets Preparation (6-8 minutes)</strong></p>
              <ul>
                <li>Audio File Preparation: Select consistent audio file for pipeline demos</li>
                <li>Pre-computed Spectrograms: Generate spectrograms to speed up demos</li>
                <li>Tensor Shape Testing: Verify 4D tensor requirements (batch, channels, freq, time)</li>
                <li>Memory Optimization: Test tensor operations with different batch sizes</li>
              </ul>

              <p><strong>Hardware and Performance Verification (4-5 minutes)</strong></p>
              <ul>
                <li>GPU Memory Check: Verify sufficient VRAM for tensor operations</li>
                <li>CUDA Performance: Benchmark tensor transfers and operations</li>
                <li>Fallback Testing: Ensure code works on CPU if GPU unavailable</li>
                <li>Memory Management: Test large tensor handling and cleanup</li>
              </ul>

              <p><strong>Presentation Materials Organization (1-2 minutes)</strong></p>
              <ul>
                <li>Slide Access: Prepare Slide 3 (tensor operations) and Slide 4 (pipeline integration)</li>
                <li>Demo Practice: Time complete pipeline execution</li>
                <li>Backup Materials: Prepare PyTorch operations examples</li>
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
                <li>GPU acceleration benefits for neural networks</li>
                <li>Seamless NumPy ↔ PyTorch conversion</li>
                <li>4D tensor shape requirements (batch, channels, height, width)</li>
                <li>Integration with Ryan and Yovannoa&apos;s PyTorch learning</li>
              </ul>

              <h4>Troubleshooting</h4>
              <ul>
                <li>If no GPU: Code works on CPU (just slower)</li>
                <li>If CUDA errors: Check PyTorch installation</li>
                <li>If memory issues: Use smaller tensors or shorter audio</li>
              </ul>

              <h4>Success Criteria</h4>
              <ul>
                <li>[ ] PyTorch setup verified with GPU detection</li>
                <li>[ ] NumPy array successfully converted to GPU tensor</li>
                <li>[ ] Complete pipeline shows all transformations</li>
                <li>[ ] GPU acceleration demonstrated within 1 minute</li>
              </ul>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </>
  );
}
