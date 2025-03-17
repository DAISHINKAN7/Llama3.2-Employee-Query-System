# ingest_csv_to_chromadb.py
import pandas as pd
import chromadb
import time
import argparse
import os
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ingest CSV data into ChromaDB')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--chroma_path', type=str, default='./chroma_db', help='Path to the ChromaDB directory')
    parser.add_argument('--collection_name', type=str, default='employee_data', help='Name of the ChromaDB collection')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for ChromaDB uploads')
    return parser.parse_args()

def create_document_from_row(row):
    """Create a document text with all details from a row"""
    return f"""Employee Record Details:
    Basic Information:
    - Requisition ID: {row['RequisitionID']}
    - Employee ID: {row['EmployeeID']}
    - Name: {row['Name']}
    - Client Name: {row['ClientName']}
    
    Dates and Timeline:
    - Joining Date: {row['JoiningDate']}
    - Day 1 Attendance: {'Marked' if row['Day1AttendanceMarked'] else 'Not Marked'}
    
    IT Asset Information:
    - Asset Status: {row['ITAssetStatus']}
    - Shipped Date: {row['ITAssetShippedDate']}
    - Delivery Date: {row['ITAssetDeliveryDate']}
    
    Shipping Details:
    - Shipping Company: {row['ShippingCompany']}
    - FedEx Tracking Number: {row['FedExTrackingNumber']}
    - FedEx Shipping Status: {row['FedExShippingStatus']}
    - FedEx Estimated Delivery: {row['FedExEstimatedDeliveryDate']}
    
    Setup and Progress:
    - Remote Setup: {'Completed' if row['RemoteSetupComplete'] else 'Pending'}
    - Progress Status: {row['ProgressStatus']}"""

def create_metadata_from_row(row):
    """Create metadata with string conversions from a row"""
    return {
        "requisition_id": str(row['RequisitionID']),
        "employee_id": str(row['EmployeeID']),
        "name": str(row['Name']),
        "joining_date": str(row['JoiningDate']),
        "it_asset_shipped_date": str(row['ITAssetShippedDate']),
        "it_asset_delivery_date": str(row['ITAssetDeliveryDate']),
        "day1_attendance_marked": str(row['Day1AttendanceMarked']),
        "it_asset_status": str(row['ITAssetStatus']),
        "shipping_company": str(row['ShippingCompany']),
        "fedex_tracking_number": str(row['FedExTrackingNumber']),
        "fedex_shipping_status": str(row['FedExShippingStatus']),
        "fedex_estimated_delivery_date": str(row['FedExEstimatedDeliveryDate']),
        "remote_setup_complete": str(row['RemoteSetupComplete']),
        "progress_status": str(row['ProgressStatus']),
        "client_name": str(row['ClientName'])
    }

def ingest_csv_to_chromadb(csv_file, chroma_path, collection_name, batch_size=100):
    """Ingest CSV data into ChromaDB"""
    # Validate the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return False

    print(f"Reading data from CSV file: {csv_file}")
    try:
        # Use pandas to read CSV
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} records from CSV")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return False

    # Setup ChromaDB client
    print(f"Setting up ChromaDB at: {chroma_path}")
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Check if collection exists
        try:
            # Try to get the collection
            collection = chroma_client.get_collection(collection_name)
            print(f"Using existing ChromaDB collection: {collection_name}")
            
            # Check if we want to clear the collection
            response = input("Collection already exists. Clear existing data? (y/n): ")
            if response.lower() == 'y':
                chroma_client.delete_collection(collection_name)
                collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Employee data with client information"}
                )
                print(f"Recreated collection: {collection_name}")
        except:
            # Create the collection if it doesn't exist
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Employee data with client information"}
            )
            print(f"Created new ChromaDB collection: {collection_name}")
    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        return False

    # Prepare documents, metadatas, and ids for ingestion
    print("Processing records and preparing for ingestion...")
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        # Create document text
        doc = create_document_from_row(row)
        
        # Create metadata
        metadata = create_metadata_from_row(row)
        
        documents.append(doc)
        metadatas.append(metadata)
        ids.append(f"doc_{idx}")

    # ChromaDB Upload
    print(f"\nUploading to ChromaDB in batches of {batch_size}...")
    upload_start_time = time.time()
    total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size else 0)
    
    for batch in tqdm(range(total_batches), desc="Uploading batches"):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(documents))
        
        collection.add(
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            ids=ids[start_idx:end_idx]
        )
    
    final_time = time.time() - upload_start_time
    
    # Print upload statistics
    print("\n=== Upload Statistics ===")
    print(f"Total Documents: {len(documents)}")
    print(f"Processing Time: {final_time:.2f} seconds")
    print(f"Average Speed: {len(documents)/final_time:.1f} docs/sec")
    
    # Save dataframe structure for reference
    df_structure = {
        "columns": list(df.columns),
        "total_records": len(df),
        "unique_clients": df['ClientName'].nunique(),
        "client_counts": df['ClientName'].value_counts().to_dict()
    }
    
    # Add collection metadata with DataFrame information
    collection.modify(metadata={
        "description": "Employee data with client information",
        "total_documents": len(documents),
        "df_structure": str(df_structure),
        "ingest_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    print(f"\nIngest complete! Data is available in ChromaDB collection: {collection_name}")
    return True

if __name__ == "__main__":
    args = parse_arguments()
    ingest_csv_to_chromadb(
        args.csv_file, 
        args.chroma_path, 
        args.collection_name,
        args.batch_size
    )