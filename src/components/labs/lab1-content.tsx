export function Lab1Content() {
  return (
    <>
      <h3>Introduction</h3>
      <p>
        This lab introduces the fundamental setup for our audio processing experiments. We will cover environment setup, basic code structure, and how to interpret initial outputs. The goal is to establish a consistent and reproducible development environment.
      </p>
      <h3>Code Snippet: Environment Check</h3>
      <p>A simple Python script to confirm that core libraries are installed and accessible.</p>
      <pre>
        <code>
{`import numpy as np
import torch

def main():
    print("--- Lab 1: Environment Check ---")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print("Environment setup is successful!")

if __name__ == "__main__":
    main()`}
        </code>
      </pre>
      <h3>Expected Output</h3>
      <p>Running the script should produce an output similar to the following, confirming your setup.</p>
      <pre>
        <code>
{`--- Lab 1: Environment Check ---
NumPy version: 1.26.4
PyTorch version: 2.3.0
Environment setup is successful!`}
        </code>
      </pre>
      <h3>Troubleshooting</h3>
      <p>
        If you encounter a <strong>ModuleNotFoundError</strong>, it means a required library is not installed. Please install it using pip:
      </p>
      <pre>
        <code>pip install numpy torch</code>
      </pre>
    </>
  );
}
