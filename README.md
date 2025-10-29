# Implementation of EEG-RAGNet: Retrieval-Augmented Graph Structure Refinement for Clinical Seizure Diagnosis

---

## 🧪 Overview

EEG-RAGNet (Retrieval-Augmented Dynamic Graph Neural Network for EEG-based Epilepsy Diagnosis) is an advanced framework for epilepsy diagnosis using EEG signals.
 It extends a Dynamic Spatial-Temporal Graph Neural Network (STGNN) architecture with **knowledge-driven refinement**.

Unlike existing graph learning-based EEG seizure detection methods, EEG-RAGNet integrates domain knowledge through a **retrieval-augmented pipeline**, allowing the model to dynamically refine its learned brain connectivity graph using biomedical knowledge extracted from epilepsy guidelines and scientific literature.


### 🔬 Key Features
- **Dynamic Graph Learning:** Learns time-varying EEG connectivity graphs via STGNN.
- **Knowledge Integration:** Aligns EEG channel semantics with biomedical knowledge embeddings.
- **RAG-Based Graph Refinement:** Refines the adjacency matrix using retrieved triplets from a medical knowledge graph.
- **Plug-and-Play Module:** Can be easily integrated into any STGNN architecture.

---

## 🧩 Repository Structure

```
.
├── EEG_Files_name.txt
├── LICENSE
├── README.md
├── args.py
├── assign_label_tags.py
├── constants.py
├── data
│   ├── __pycache__
│   ├── clip_resample_signals.py
│   ├── constants.py
│   ├── data_utils.py
│   ├── dataloader_chb.py
│   ├── dataloader_detection.py
│   ├── dataloader_detection_LLM.py
│   ├── dataloader_prediction.py
│   ├── electrode_graph
│   ├── file_markers_detection
│   ├── file_markers_prediction
│   ├── preprocess_detection.py
│   ├── preprocess_prediction.py
│   └── resample_signals.py
├── knowledge.json
├── label.txt
├── main.py
├── model
│   ├── BIOT.py
│   ├── DCRNN.py
│   ├── EGCN.py
│   ├── EvoBrain.py
│   ├── cell.py
│   ├── cnnlstm.py
│   ├── dense_inception
│   ├── eeg_ragnet
│   ├── graph_constructor.py
│   ├── loss.py
│   └── lstm.py
├── processed_data
│   ├── 00000906_s007_t000.h5
│   ├── 00000906_s007_t002.h5
│   ├── 00000906_s007_t003.h5
│   └── How_to_Access_TUSZ_dataset.md
├── requirements.txt
└── utils.py
```


---

## ⚙️ Installation

### 1. Clone the repository
```
git clone https://github.com/<your-org>/EEG-RAGNet.git
cd EEG-RAGNet
```

### 2. Create environment
```
conda create -n eeg_ragnet python=3.9
conda activate eeg_ragnet
pip install -r requirements.txt
```

### 3. Prepare data

Place your pre-processed EEG `.h5` files under. NOTE that full TUSZ dataset can only be accessed by request:
```
data/processed_data/
```

Place the corresponding label files under:
```
data/file_markers_detection/

```

If you have not yet generated the knowledge base:
```
python model/eeg_ragnet/knowledge_base.py

```
This will produce `knowledge.json` and `KG_triplets.json`.



## 🤔 How It Works

EEG-RAGNet introduces a three-stage refinement loop inside the STGNN pipeline:

1. **Semantic Query Projection (`semantic_query.py`)**
   EEG node features → semantic embeddings `Q_t`.

2. **Knowledge Retrieval (`faiss_retriever.py`)**
   Retrieve top-K relevant biomedical triplets from FAISS index.

3. **Graph Refinement (`graph_refiner.py`)**
   Fuse STGNN-learned adjacency `A_t` with retrieved semantic graph `A_rag`:
   [
   A_{\text{refined}} = \sigma(\alpha A_t + (1-\alpha)A_{\text{rag}})
   ]

---



### Training

```
python main.py --model_name evolvegcn --use_ragnet
```

### Important Arguments

| Argument             | Description                      | Default                        |
| -------------------- | -------------------------------- | ------------------------------ |
| `--use_ragnet`       | Enable RAG-based refinement      | `False`                        |
| `--kg_triplets_path` | Path to knowledge graph triplets | `./knowledge/kg_triplets.json` |
| `--faiss_index_path` | Path to FAISS index              | `./knowledge/faiss.index`      |
| `--refine_threshold` | Edge confidence threshold        | `0.6`                          |
| `--refine_alpha`     | Fusion coefficient               | `0.7`                          |
| `--refine_interval`  | Perform refinement every N steps | `1`                            |

For all configuration options, see [`args.py`](./args.py).

---

##  Example Training Workflow

1. **Build knowledge graph**

   ```
   python model/eeg_ragnet/knowledge_base.py
   ```
2. **Train model**

   ```
   python main.py --model_name evolvegcn --use_ragnet --epochs 100
   ```
3. **View refinement results**
   Refined adjacency matrices and triplet logs are saved in:

   ```
   results/graph_refinement/
   ```

---

### Outputs

* Logs of retrieved biomedical knowledge supporting each refined edge.
* Knowledge-enhanced graph adjacency matrix used for downstream classification.

---

## Citation

If you find this project useful, please considering cite our work:

```
@article{EEGRAGNet2025,
  title={EEG-RAGNet: Retrieval-Augmented Graph Neural Networks for Knowledge-Guided EEG Epilepsy Diagnosis},
  author={First Author, Second Author, Third Author, et al.},
  journal={Under Review},
  year={2025}
}
```




