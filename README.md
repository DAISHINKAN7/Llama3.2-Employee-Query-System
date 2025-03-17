# Employee Data Analysis System

A two-component system for analyzing employee onboarding data with ChromaDB and LLM-based insights.

---

## 📋 Overview

This system consists of two separate but interconnected components:

1. **Data Ingestion Script** (`ingest_csv_to_chromadb.py`)
   - Loads employee data from CSV files
   - Processes and stores records in ChromaDB
   - Creates vector embeddings for semantic search

2. **Analysis Application** (`app.py`)
   - Streamlit-based user interface
   - Connects to the pre-populated ChromaDB
   - Uses LM Studio for advanced data analysis
   - Provides insights, recommendations, and visualizations

---

## 🚀 Getting Started

### Prerequisites

#### Software Requirements

- Python 3.8+
- LM Studio with a local server running

#### Python Packages

```bash
pip install streamlit pandas chromadb requests tqdm
```

#### Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv chromadb_env

# Activate on Windows
chromadb_env\Scripts\activate

# Activate on macOS/Linux
source chromadb_env/bin/activate

# Install required packages
pip install streamlit pandas chromadb requests tqdm
```

---

## 🔄 Execution Sequence

> **IMPORTANT:** These components must be run in the correct order:
> 1. **FIRST:** Run the ingestion script (`ingest_csv_to_chromadb.py`) to populate ChromaDB
> 2. **SECOND:** Run the analysis application (`app.py`) to query and analyze the data
>
> **Never run the application without first populating ChromaDB with the ingestion script.**

### Step 1: Prepare Your Data

Ensure your CSV file contains these columns:
- `RequisitionID`
- `EmployeeID`
- `Name`
- `ClientName`
- `JoiningDate`
- `Day1AttendanceMarked`
- `ITAssetStatus`
- `ITAssetShippedDate`
- `ITAssetDeliveryDate`
- `ShippingCompany`
- `FedExTrackingNumber`
- `FedExShippingStatus`
- `FedExEstimatedDeliveryDate`
- `RemoteSetupComplete`
- `ProgressStatus`

### Step 2: Set Up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Launch LM Studio and load a model (recommended: `llama-3.2-3b-instruct`)
3. Start the local server (default URL: `http://localhost:1234`)

### Step 3: Run the Ingestion Script FIRST

**⚠️ Always run this script first before starting the application**

```bash
python ingest_csv_to_chromadb.py --csv_file your_employee_data.csv
```

This script must complete successfully before proceeding to Step 4. The ChromaDB database must be populated with your data or the application will have nothing to analyze.

#### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--csv_file` | Path to your CSV file | (Required) |
| `--chroma_path` | Directory for ChromaDB | `./chroma_db` |
| `--collection_name` | Name of ChromaDB collection | `employee_data` |
| `--batch_size` | Records per batch | `100` |

### Step 4: Launch the Analysis Application SECOND

**⚠️ Only run this after the ingestion script has successfully completed**

```bash
streamlit run app.py
```

The application cannot function without the pre-populated ChromaDB that was created in Step 3.

### Step 5: Use the Application

1. Enter the LM Studio URL (default: `http://localhost:1234`)
2. Click "Initialize System" in the sidebar
3. Begin asking questions about your employee data

---

## 🔍 System Architecture

```
┌─────────────┐     ┌───────────────────┐     ┌─────────────┐
│  CSV File   │────▶│  Ingestion Script  │────▶│  ChromaDB   │
└─────────────┘     └───────────────────┘     └──────┬──────┘
                                                     │
                                                     ▼
┌─────────────┐     ┌───────────────────┐     ┌─────────────┐
│   Results   │◀────│    LM Studio      │◀────│ Streamlit   │
└─────────────┘     └───────────────────┘     └─────────────┘
```

---

## ❓ Example Queries

- "What is the average delivery time for IT assets? Break down by client."
- "Show me a detailed analysis of all in-transit assets and their current status."
- "Which clients have the highest and lowest setup completion rates?"
- "Analyze the onboarding timeline trends across all clients."
- "What are the common shipping delays and their patterns?"
- "Compare remote setup completion rates between different clients."
- "Give me a full analysis of asset delivery performance by shipping company."

---

## ⚠️ Troubleshooting

### Column Name Issues

If you see errors about missing columns:
- Check the "Available columns" debug output in the app
- Verify your CSV column names match the expected names
- The app will try to find alternative column names, but direct matches are best

### ChromaDB Connection Issues

- Ensure both scripts use the same ChromaDB path
- Check that the collection name matches (`employee_data` by default)
- Verify the ingestion script completed successfully

### LM Studio Connection Issues

- Confirm LM Studio is running with a model loaded
- Check the local server URL (usually `http://localhost:1234`)
- Verify your model supports the chat completions API

---

## 🗂️ File Descriptions and Run Order

### 1️⃣ `ingest_csv_to_chromadb.py` (RUN FIRST)

- Standalone script for data ingestion
- Reads CSV data and creates ChromaDB collection
- Processes document text and metadata for each record
- **Must be run successfully before launching the app**

### 2️⃣ `app.py` (RUN SECOND)

- Streamlit application for data analysis
- Connects to pre-populated ChromaDB created by the ingestion script
- Provides query interface and visualizations
- Communicates with LM Studio for analysis
- **Depends on the ChromaDB created by the ingestion script**

---

## 📊 Data Flow Details

1. **CSV → ChromaDB**
   - CSV records are read and processed
   - Each record becomes a document in ChromaDB
   - Field values are stored as metadata
   - Document text contains formatted employee record details

2. **ChromaDB → Analysis**
   - Application loads data from ChromaDB into a DataFrame
   - Query system performs statistical analysis on the data
   - Relevant documents are retrieved based on query similarity
   - LM Studio combines document context with analysis for insights