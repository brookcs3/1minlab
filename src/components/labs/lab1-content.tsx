import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Code, MessageSquare, Terminal, Users, Info, CheckCircle, AlertTriangle, Wrench } from "lucide-react";

export function Lab1Content() {
  return (
    <>
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold font-headline">
              Lab 1: Data Integration and Preparation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              Let's crush this 1-minute presentation! We're going to live-code our way through audio loading and visualization, showing how audio becomes numerical data for machine learning. It's fast, it's live, and it demonstrates the foundation of our entire audio pipeline!
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base font-semibold font-headline">Timing: 0:00 - 0:15 (15 seconds): Introduction & Load Audio</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
              <pre><code>{`import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
audio, sr = librosa.load('sample_audio.wav', sr=22050)

# Display basic information
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")`}</code></pre>
            </div>
            <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
              <p className="pl-6 border-l-2 border-accent ml-2">"Welcome to Lab 1! We start by loading audio into Python using librosa. This transforms any audio file into a NumPy array of numbers."</p>
            </div>
            <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><Terminal size={16} /> Expected Output:</h4>
              <pre><code>{`Audio shape: (661500,)
Sample rate: 22050 Hz
Duration: 30.00 seconds`}</code></pre>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base font-semibold font-headline">Timing: 0:15 - 0:35 (20 seconds): Visualize Waveform</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
              <pre><code>{`# Visualize first 1000 samples
plt.figure(figsize=(12, 4))
plt.plot(audio[:1000])
plt.title("Audio Waveform (first 1000 samples)")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.show()`}</code></pre>
            </div>
            <div>
                <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
                <p className="pl-6 border-l-2 border-accent ml-2">"See this waveform? Each point represents the air pressure at a specific moment. The oscillations show the sound wave pattern."</p>
            </div>
             <div>
                <h4 className="font-semibold flex items-center gap-2 mb-2"><Info size={16} /> Key Observation:</h4>
                <p className="pl-6 border-l-2 border-accent ml-2">Point out the waveform oscillations and amplitude range.</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base font-semibold font-headline">Timing: 0:35 - 0:50 (15 seconds): Spectrogram Preview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><Code size={16} /> Code to Run:</h4>
              <pre><code>{`# Preview spectrogram transformation
spectrogram = librosa.stft(audio)
magnitude = np.abs(spectrogram)

print(f"Waveform shape: {audio.shape}")
print(f"Spectrogram shape: {magnitude.shape}")
print(f"Frequency bins: {magnitude.shape[0]}")
print(f"Time frames: {magnitude.shape[1]}")`}</code></pre>
            </div>
            <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
              <p className="pl-6 border-l-2 border-accent ml-2">"Here's a preview of Lab 3: STFT transforms our 1D waveform into a 2D spectrogram. This is what neural networks process!"</p>
            </div>
            <div>
              <h4 className="font-semibold flex items-center gap-2 mb-2"><Terminal size={16} /> Expected Output:</h4>
              <pre><code>{`Waveform shape: (661500,)
Spectrogram shape: (1025, 646)
Frequency bins: 1025
Time frames: 646`}</code></pre>
            </div>
          </CardContent>
        </Card>
        
        <Card>
            <CardHeader>
                <CardTitle className="text-base font-semibold font-headline">Timing: 0:50 - 1:00 (10 seconds): Wrap-up</CardTitle>
            </CardHeader>
            <CardContent>
                <h4 className="font-semibold flex items-center gap-2 mb-2"><MessageSquare size={16} /> Narration:</h4>
                <p className="pl-6 border-l-2 border-accent ml-2">"That's Lab 1! Audio is just numerical data. This foundation enables everything from NumPy processing to U-Net separation."</p>
            </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Users size={20} className="text-primary"/>
              <CardTitle className="text-base font-semibold font-headline">Team Member Connections:</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="list-disc pl-5 space-y-2">
              <li><strong>Cameron's POC:</strong> Starts with loading "Intergalactic" audio file</li>
              <li><strong>Yovannoa's Classifier:</strong> Demonstrated dataset loading and transformation</li>
              <li><strong>cervanj2's Architecture:</strong> This is the foundation of the Preprocessing Layer</li>
            </ul>
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
                <h4>Setup Beforehand (15-20 minutes total prep time)</h4>
                <p>Each team member will assemble their own code and working repository for this lab.</p>

                <p><strong>Environment Setup (5-7 minutes)</strong></p>
                <ul>
                  <li>Install Dependencies: <code>pip install librosa numpy matplotlib</code></li>
                  <li>Verify Installation: Test imports in Python environment</li>
                  <li>IDE Setup: Prepare Jupyter notebook or Python IDE with code execution capabilities</li>
                </ul>

                <p><strong>Code Assembly and Repository Setup (7-10 minutes)</strong></p>
                <ul>
                  <li>Create Working Directory: Set up dedicated folder for Lab 1</li>
                  <li>Assemble Code Files:
                    <ul>
                      <li>Create <code>lab1_basic.py</code> with functions: <code>load_audio_basic()</code>, <code>visualize_waveform()</code>, <code>spectrogram_preview()</code></li>
                      <li>Copy/paste from provided code examples or write from scratch</li>
                      <li>Add proper imports and docstrings</li>
                    </ul>
                  </li>
                  <li>Test Code Independently: Run functions with sample data to ensure they work</li>
                  <li>Version Control: Initialize git repo and commit working code</li>
                </ul>

                <p><strong>Files and Assets Preparation (3-5 minutes)</strong></p>
                <ul>
                  <li>Audio File Acquisition: Obtain or download <code>sample_audio.wav</code> (30-second clip recommended)</li>
                  <li>Alternative Audio Sources: Prepare fallback options (librosa examples, personal audio files)</li>
                  <li>Test File Loading: Verify audio file loads correctly in your environment</li>
                </ul>

                <p><strong>Hardware and Display Verification (2-3 minutes)</strong></p>
                <ul>
                  <li>Display Setup: Confirm matplotlib plots render properly</li>
                  <li>Audio Playback: Optional - test audio playback for verification</li>
                  <li>Performance Check: Time loading of sample audio file</li>
                </ul>

                <p><strong>Presentation Materials Organization (1-2 minutes)</strong></p>
                <ul>
                  <li>Slide Preparation: Ensure access to Slide 3 (code example) and Slide 4 (spectrogram preview)</li>
                  <li>Backup Materials: Prepare Slide 2 if time allows for audio concepts explanation</li>
                  <li>Timing Practice: Run through demo script once to verify 1-minute timing</li>
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
                  <li>Audio as numerical arrays (not "magic")</li>
                  <li>Sample rate and duration concepts</li>
                  <li>Waveform visualization</li>
                  <li>Preview of spectrogram transformation</li>
                  <li>Connection to the POC starting point</li>
                </ul>

                <h4 className="flex items-center gap-2 mt-4"><AlertTriangle size={16} /> Troubleshooting</h4>
                <ul>
                  <li>If audio file missing: Use <code>librosa.example('nutcracker')</code> as fallback</li>
                  <li>If plots don't show: Ensure matplotlib backend is configured</li>
                  <li>If slow loading: Use shorter audio file or duration parameter</li>
                </ul>

                <h4 className="flex items-center gap-2 mt-4"><CheckCircle size={16} /> Success Criteria</h4>
                <ul>
                  <li>[ ] Audio loaded successfully</li>
                  <li>[ ] Basic info printed (shape, rate, duration)</li>
                  <li>[ ] Waveform plot displayed</li>
                  <li>[ ] Spectrogram preview shows transformation</li>
                  <li>[ ] Concepts explained clearly in 1 minute</li>
                </ul>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </>
  );
}
