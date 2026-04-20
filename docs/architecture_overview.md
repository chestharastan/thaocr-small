# ThaoNet Architecture Overview

The following diagram illustrates the data flow and components of the **ThaoNet** Recognition Model.

```mermaid
graph LR
    subgraph Input [1. Input Processing]
        IMG(Input Image) -->|Resize + Gray| PRE[64 x W]
    end

    subgraph Backbone [2. Feature Extractor (ResNet)]
        PRE -->|Conv Stride 2| B1[Layer1 (32xW/2)]
        B1 -->|Conv Stride 2| B2[Layer2 (16xW/4)]
        B2 -->|Conv Stride 2| B3[Layer3 (8xW/4)]
        B3 -->|Conv Stride 2| B4[Layer4 (4xW/4)]
    end

    subgraph Neck [3. Sequence Bridge]
        B4 -->|Permute & Flatten| FEAT[Sequence T x 1024]
        FEAT -->|Linear + LayerNorm| EMBD[Embeddings T x 256]
    end

    subgraph Head [4. Context Modeling (Transformer)]
        EMBD -->|Positional Encoding| POS[Add Sine Pos]
        POS -->|Self-Attention| TR1[Encoder Layer 1]
        TR1 -->|Feed Forward| TR2[Encoder Layer 2]
        TR2 -->|...| TRN[Encoder Layer N]
    end

    subgraph Decode [5. Prediction]
        TRN -->|Linear Classifier| LOGITS[Logits (T x Vocab)]
        LOGITS -->|CTC Decode| TEXT[Final Text String]
    end
    
    style Backbone fill:#e1f5fe,stroke:#01579b
    style Head fill:#f3e5f5,stroke:#4a148c
    style Decode fill:#e8f5e9,stroke:#1b5e20
```

## Component Details

### 1. Backbone (ResNet)
*   **Purpose**: Extracts visual features (shapes, curves, strokes).
*   **Operation**: Reduces image height by **16x** and width by **4x**.
*   **Output**: A 3D tensor of features `[Channels, H/16, W/4]`.

### 2. Neck (Linear)
*   **Purpose**: Converts 2D image features into a 1D sequence for the Transformer.
*   **Operation**: Collapses the vertical dimension (Height) into the channel dimension.
*   **Output**: A sequence of vectors `[TimeSteps, HiddenDim]`.

### 3. Head (Transformer)
*   **Purpose**: Understands character context (e.g., "independent vowel" vs "dependent vowel").
*   **Mechanism**: Uses **Self-Attention** to look at the entire sequence at once.
*   **Output**: Probability distribution over the vocabulary for each time step.

### 4. Decode (CTC)
*   **Purpose**: Aligns the sequence of probabilities to the final text.
*   **Mechanism**: Removes duplicate characters and blanks (`-`) to form the final word.
