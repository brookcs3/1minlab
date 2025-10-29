export function Lab5Content() {
  return (
    <>
      <h3>The U-Net Architecture</h3>
      <p>
        U-Net is a convolutional neural network (CNN) architecture originally developed for biomedical image segmentation. Its distinctive U-shape, consisting of a contracting path (encoder) and an expansive path (decoder) with skip connections, makes it highly effective for tasks requiring precise localization, such as audio source separation from a spectrogram.
      </p>
      <h3>Core Building Block: Double Convolution</h3>
      <p>
        The encoder and decoder are built from repeating blocks. A common block is a sequence of two 3x3 convolutions, each followed by Batch Normalization and a ReLU activation.
      </p>
      <pre>
        <code>
{`import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# This block can be used to build the full U-Net
print("DoubleConv block defined.")
`}
        </code>
      </pre>
      <h3>Integration for Audio Separation</h3>
      <p>
        In a complete pipeline, the U-Net takes a mixed audio spectrogram as input. It is trained to output a soft mask, which, when multiplied with the input spectrogram, isolates a specific source (e.g., vocals). The final step is to apply an inverse STFT to the masked spectrogram to recover the separated audio waveform.
      </p>
    </>
  );
}
