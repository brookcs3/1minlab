import { AudioWaveform, BarChart3, FileText, Flame } from "lucide-react";
import { UNetIcon } from "@/components/icons";
import { LabCard } from "@/components/lab-card";
import { Lab1Content } from "@/components/labs/lab1-content";
import { Lab2Content } from "@/components/labs/lab2-content";
import { Lab3Content } from "@/components/labs/lab3-content";
import { Lab4Content } from "@/components/labs/lab4-content";
import { Lab5Content } from "@/components/labs/lab5-content";

export default function Home() {
  return (
    <div className="flex min-h-screen w-full flex-col bg-background">
      <header className="sticky top-0 z-10 border-b bg-card/80 backdrop-blur-sm">
        <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
          <h1 className="font-headline text-2xl font-bold">1 Min Demos</h1>
          <p className="text-sm text-muted-foreground">Audio & Deep Learning</p>
        </div>
      </header>
      <main className="flex-1">
        <div className="container mx-auto px-4 py-8 md:px-6 md:py-12">
          <div className="mx-auto grid max-w-4xl grid-cols-1 gap-8">
            <LabCard
              labNumber={1}
              title="Project Setup"
              description="Environment configuration, code structure, and basic I/O."
              icon={<FileText className="h-6 w-6" />}
            >
              <Lab1Content />
            </LabCard>
            <LabCard
              labNumber={2}
              title="NumPy for Audio"
              description="Audio processing with NumPy, volume operations, and analysis."
              icon={<AudioWaveform className="h-6 w-6" />}
            >
              <Lab2Content />
            </LabCard>
            <LabCard
              labNumber={3}
              title="STFT with Librosa"
              description="Spectrograms, visualization, and round-trip reconstruction."
              icon={<BarChart3 className="h-6 w-6" />}
            >
              <Lab3Content />
            </LabCard>
            <LabCard
              labNumber={4}
              title="Intro to PyTorch"
              description="Setup checks, NumPy-to-tensor conversions, and pipeline integration."
              icon={<Flame className="h-6 w-6" />}
            >
              <Lab4Content />
            </LabCard>
            <LabCard
              labNumber={5}
              title="U-Net for Separation"
              description="U-Net architecture and complete integration for audio separation."
              icon={<UNetIcon className="h-6 w-6" />}
            >
              <Lab5Content />
            </LabCard>
          </div>
        </div>
      </main>
      <footer className="border-t">
        <div className="container mx-auto px-4 py-6 md:px-6">
          <p className="text-center text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} Lab Write-Ups. All Rights Reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
