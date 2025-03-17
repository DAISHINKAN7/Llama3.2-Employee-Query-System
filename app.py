# app.py
import streamlit as st
import pandas as pd
import chromadb
import requests
import json
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Employee Data Analysis System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class EmployeeDataQuerySystem:
    def __init__(self, lmstudio_url):
        """Initialize the system with direct ChromaDB access"""
        self.base_url = lmstudio_url
        self.chroma_client = None
        self.collection = None
        self.total_documents = 0
        self.dataframe = None
        
        # Configure retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @st.cache_data(ttl=3600)
    def query_lmstudio(_self, prompt, max_tokens=2000):
        """Enhanced LM Studio query with better token handling"""
        try:
            formatted_prompt = "Analyze this data comprehensively and provide a detailed response. Focus on specific details and numbers.\n\n" + prompt + "\n\nRemember to:\n1. Provide specific examples and numbers\n2. Include relevant dates\n3. Mention client-specific details\n4. Show any relevant statistics"
            
            # Limit prompt size
            max_length = 8000
            if len(formatted_prompt) > max_length:
                formatted_prompt = formatted_prompt[:max_length] + "\n...[Additional context available]"
            
            headers = {"Content-Type": "application/json"}
            data = {
                "messages": [{"role": "user", "content": formatted_prompt}],
                "model": "llama-3.2-3b-instruct",
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            for attempt in range(3):
                try:
                    with st.spinner(f'🤔 Processing... (Attempt {attempt + 1}/3)'):
                        response = _self.session.post(
                            f"{_self.base_url}/v1/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=(30, 180)
                        )
                        response.raise_for_status()
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content']
                            if content and len(content.strip()) > 0:
                                return content
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            
            return None
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return None

    def setup_chromadb(self):
        """Connect to existing ChromaDB instance"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            try:
                # Try to get existing collection
                self.collection = self.chroma_client.get_collection("employee_data")
                
                # Get the count of documents
                collection_data = self.collection.get()
                self.total_documents = len(collection_data['ids']) if 'ids' in collection_data else 0
                
                st.success(f"✓ Connected to ChromaDB collection with {self.total_documents} records")
                
                # Load data from ChromaDB into a DataFrame for analysis
                self.load_dataframe_from_chromadb()
                
            except Exception as e:
                st.error(f"ChromaDB collection error: {str(e)}")
                raise Exception("Could not access employee_data collection in ChromaDB. Make sure it's populated using the ingestion script.")
                
        except Exception as e:
            st.error(f"🚫 Error setting up ChromaDB: {str(e)}")
            raise

    def load_dataframe_from_chromadb(self):
        """Load data from ChromaDB into a DataFrame for analysis"""
        try:
            # Get all documents from ChromaDB
            collection_data = self.collection.get()
            
            if not collection_data or 'metadatas' not in collection_data or not collection_data['metadatas']:
                raise Exception("No data found in ChromaDB collection")
            
            # Create DataFrame from metadatas
            df_data = []
            
            # Process metadata entries
            for metadata in collection_data['metadatas']:
                # Convert all field names to standard column names (capitalized)
                record = {}
                
                # Map lowercase metadata keys to capitalized column names
                field_mapping = {
                    'requisitionid': 'RequisitionID', 
                    'employeeid': 'EmployeeID',
                    'name': 'Name',
                    'clientname': 'ClientName',
                    'joiningdate': 'JoiningDate',
                    'day1attendancemarked': 'Day1AttendanceMarked',
                    'itassetstatus': 'ITAssetStatus',
                    'itassetshippeddate': 'ITAssetShippedDate',
                    'itassetdeliverydate': 'ITAssetDeliveryDate',
                    'shippingcompany': 'ShippingCompany',
                    'fedextracingnumber': 'FedExTrackingNumber',
                    'fedexshippingstatus': 'FedExShippingStatus',
                    'fedexestimateddeliverydate': 'FedExEstimatedDeliveryDate',
                    'remotesetupcomplete': 'RemoteSetupComplete',
                    'progressstatus': 'ProgressStatus'
                }
                
                # Create a record with properly capitalized field names
                for key, value in metadata.items():
                    if key in field_mapping:
                        record[field_mapping[key]] = value
                    else:
                        # Keep original key if not in mapping
                        record[key] = value
                
                df_data.append(record)
            
            # Create DataFrame
            self.dataframe = pd.DataFrame(df_data)
            
            # Print column names for debugging
            st.write("Available columns:", list(self.dataframe.columns))
            
            # Ensure required columns exist - use lowercase alternatives if standard names don't exist
            required_columns = {
                'ClientName': ['clientname', 'client_name', 'client'],
                'RemoteSetupComplete': ['remotesetupcomplete', 'remote_setup_complete', 'setupcomplete'],
                'ITAssetStatus': ['itassetstatus', 'asset_status', 'status'],
                'JoiningDate': ['joiningdate', 'joining_date', 'date_joined'],
                'ITAssetShippedDate': ['itassetshippeddate', 'shipped_date'],
                'ITAssetDeliveryDate': ['itassetdeliverydate', 'delivery_date'],
                'Day1AttendanceMarked': ['day1attendancemarked', 'attendance_marked']
            }
            
            # Add missing columns with default values or rename existing columns
            for standard_name, alternatives in required_columns.items():
                if standard_name not in self.dataframe.columns:
                    # Check if any alternative exists
                    found = False
                    for alt in alternatives:
                        if alt in self.dataframe.columns:
                            self.dataframe[standard_name] = self.dataframe[alt]
                            found = True
                            break
                    
                    # If no alternative found, add empty column
                    if not found:
                        st.warning(f"Column '{standard_name}' not found in data. Using placeholder values.")
                        if standard_name == 'ClientName':
                            self.dataframe[standard_name] = 'Unknown Client'
                        elif standard_name in ['RemoteSetupComplete', 'Day1AttendanceMarked']:
                            self.dataframe[standard_name] = False
                        elif 'Date' in standard_name:
                            self.dataframe[standard_name] = pd.NaT
                        else:
                            self.dataframe[standard_name] = 'Unknown'
            
            # Convert boolean fields from strings
            if 'RemoteSetupComplete' in self.dataframe.columns:
                # Convert 'True'/'False' strings to boolean
                self.dataframe['RemoteSetupComplete'] = self.dataframe['RemoteSetupComplete'].map(
                    lambda x: True if str(x).lower() == 'true' else False if str(x).lower() == 'false' else bool(x) if isinstance(x, bool) else False
                )
                
            if 'Day1AttendanceMarked' in self.dataframe.columns:
                # Convert 'True'/'False' strings to boolean
                self.dataframe['Day1AttendanceMarked'] = self.dataframe['Day1AttendanceMarked'].map(
                    lambda x: True if str(x).lower() == 'true' else False if str(x).lower() == 'false' else bool(x) if isinstance(x, bool) else False
                )
            
            # Show a sample of the data that was loaded
            st.success(f"✓ Created DataFrame with {len(self.dataframe)} records from ChromaDB")
            
        except Exception as e:
            st.error(f"Error loading data from ChromaDB: {str(e)}")
            raise

    def convert_to_serializable(self, obj):
        """Convert NumPy data types to Python standard types for JSON serialization"""
        import numpy as np
        
        # Use generic NumPy type categories for forward compatibility
        if isinstance(obj, np.integer):  # Generic check for all NumPy integer types
            return int(obj)
        elif isinstance(obj, np.floating):  # Generic check for all NumPy float types
            return float(obj)
        elif isinstance(obj, np.bool_):  # Boolean type
            return bool(obj)
        elif isinstance(obj, np.ndarray):  # NumPy arrays
            return obj.tolist()
        elif isinstance(obj, dict):  # Recursively handle dictionaries
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):  # Recursively handle lists
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj  # Return everything else unchanged

    def analyze_full_dataset(self, question):
        """Comprehensive dataset analysis"""
        try:
            # Convert dates for analysis
            date_cols = ['JoiningDate', 'ITAssetShippedDate', 'ITAssetDeliveryDate']
            for col in date_cols:
                if col in self.dataframe.columns:
                    self.dataframe[col] = pd.to_datetime(self.dataframe[col], errors='coerce')
            
            # Calculate key metrics
            metrics = {
                "total_records": len(self.dataframe),
                "unique_clients": self.dataframe['ClientName'].nunique(),
                "completed_setups": self.dataframe['RemoteSetupComplete'].fillna(False).astype(bool).sum(),
                "in_transit_assets": len(self.dataframe[
                    self.dataframe['ITAssetStatus'].str.contains('transit', case=False, na=False)
                ]),
            }
            
            # Calculate average delivery days if shipping and delivery dates are available
            if 'ITAssetShippedDate' in self.dataframe.columns and 'ITAssetDeliveryDate' in self.dataframe.columns:
                # Only for rows with both dates valid
                valid_delivery_rows = self.dataframe.dropna(subset=['ITAssetShippedDate', 'ITAssetDeliveryDate'])
                if len(valid_delivery_rows) > 0:
                    metrics["avg_delivery_days"] = (
                        valid_delivery_rows['ITAssetDeliveryDate'] - 
                        valid_delivery_rows['ITAssetShippedDate']
                    ).dt.days.mean()
            
            # Client-specific analysis
            client_analysis = {}
            for client in self.dataframe['ClientName'].unique():
                client_data = self.dataframe[self.dataframe['ClientName'] == client]
                
                client_metrics = {
                    "total_employees": len(client_data),
                    "completed_setups": client_data['RemoteSetupComplete'].fillna(False).astype(bool).sum(),
                    "setup_completion_rate": (client_data['RemoteSetupComplete'].fillna(False).astype(bool).sum() / len(client_data)) * 100 if len(client_data) > 0 else 0,
                    "in_transit": len(client_data[
                        client_data['ITAssetStatus'].str.contains('transit', case=False, na=False)
                    ]),
                }
                
                # Add delivery time if available
                if 'ITAssetShippedDate' in client_data.columns and 'ITAssetDeliveryDate' in client_data.columns:
                    valid_client_delivery = client_data.dropna(subset=['ITAssetShippedDate', 'ITAssetDeliveryDate'])
                    if len(valid_client_delivery) > 0:
                        client_metrics["avg_delivery_time"] = (
                            valid_client_delivery['ITAssetDeliveryDate'] - 
                            valid_client_delivery['ITAssetShippedDate']
                        ).dt.days.mean()
                
                client_analysis[client] = client_metrics
            
            # Recent trends (past 30 days)
            try:
                if 'JoiningDate' in self.dataframe.columns:
                    recent_data = self.dataframe[
                        self.dataframe['JoiningDate'] > 
                        (pd.Timestamp.now() - pd.Timedelta(days=30))
                    ]
                    
                    recent_metrics = {
                        "recent_joiners": len(recent_data),
                        "recent_completed_setups": recent_data['RemoteSetupComplete'].fillna(False).astype(bool).sum(),
                        "recent_setup_completion_rate": (recent_data['RemoteSetupComplete'].fillna(False).astype(bool).sum() / len(recent_data)) * 100 if len(recent_data) > 0 else 0,
                        "recent_in_transit": len(recent_data[
                            recent_data['ITAssetStatus'].str.contains('transit', case=False, na=False)
                        ])
                    }
                else:
                    recent_metrics = {"recent_data_unavailable": True}
            except Exception as e:
                st.warning(f"Could not calculate recent trends: {str(e)}")
                recent_metrics = {"recent_data_error": str(e)}
            
            # Shipping company performance
            shipping_performance = {}
            if 'ShippingCompany' in self.dataframe.columns:
                for company in self.dataframe['ShippingCompany'].dropna().unique():
                    company_data = self.dataframe[self.dataframe['ShippingCompany'] == company]
                    
                    if len(company_data) > 0 and 'ITAssetShippedDate' in company_data.columns and 'ITAssetDeliveryDate' in company_data.columns:
                        valid_company_delivery = company_data.dropna(subset=['ITAssetShippedDate', 'ITAssetDeliveryDate'])
                        
                        if len(valid_company_delivery) > 0:
                            shipping_performance[company] = {
                                "total_shipments": len(company_data),
                                "avg_delivery_days": (
                                    valid_company_delivery['ITAssetDeliveryDate'] - 
                                    valid_company_delivery['ITAssetShippedDate']
                                ).dt.days.mean(),
                                "on_time_delivery_rate": len(valid_company_delivery[
                                    valid_company_delivery['ITAssetDeliveryDate'] <= 
                                    valid_company_delivery['FedExEstimatedDeliveryDate']
                                ]) / len(valid_company_delivery) * 100 if 'FedExEstimatedDeliveryDate' in valid_company_delivery.columns and len(valid_company_delivery) > 0 else None
                            }
            
            return {
                "overall_metrics": metrics,
                "client_analysis": client_analysis,
                "recent_trends": recent_metrics,
                "shipping_performance": shipping_performance
            }
            
        except Exception as e:
            st.error(f"Error in dataset analysis: {str(e)}")
            return None

    def generate_structured_recommendations(self, analysis_text):
        """Generate structured recommendations from analysis text"""
        try:
            # Prepare a prompt for extracting structured recommendations
            recommendation_prompt = "Based on this analysis, extract 3-5 specific, actionable recommendations.\nFor each recommendation, provide:\n1. Category (Shipping, IT Asset Management, Onboarding, etc.)\n2. Priority level (1-5, with 5 being highest)\n3. The specific recommendation (be detailed and actionable)\n4. Impact area (which metrics will improve)\n5. Implementation difficulty (1-5, with 5 being hardest)\n6. Estimated improvement (percentage or qualitative assessment)\n\nFormat as a JSON array of objects. Example:\n[\n    {\n        \"category\": \"IT Asset Management\",\n        \"priority\": 4,\n        \"recommendation\": \"Implement pre-shipping verification checklist to reduce misconfigured assets\",\n        \"impact_area\": \"Setup completion rate, employee satisfaction\",\n        \"implementation_difficulty\": 2,\n        \"estimated_improvement\": 15.0\n    },\n    ...\n]\n\nAnalysis: " + analysis_text
            
            # Get structured recommendations
            recommendations_json = self.query_lmstudio(recommendation_prompt, max_tokens=1500)
            
            if not recommendations_json:
                return []
                
            # Extract JSON from response
            try:
                # Find JSON array in the response
                start_idx = recommendations_json.find('[')
                end_idx = recommendations_json.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = recommendations_json[start_idx:end_idx]
                    recommendations = json.loads(json_str)
                    
                    # Validate structure
                    valid_recommendations = []
                    for rec in recommendations:
                        if isinstance(rec, dict) and 'recommendation' in rec:
                            valid_recommendations.append(rec)
                    
                    return valid_recommendations
                else:
                    return []
            except Exception as e:
                st.warning(f"Could not parse recommendations JSON: {str(e)}")
                return []
                
        except Exception as e:
            st.error(f"Error generating structured recommendations: {str(e)}")
            return []

    def query(self, question):
        """Process query with comprehensive analysis"""
        try:
            with st.spinner("🔍 Analyzing complete dataset..."):
                # Get comprehensive analysis
                analysis = self.analyze_full_dataset(question)
                if not analysis:
                    raise Exception("Failed to analyze dataset")
                
                # Get relevant documents
                results = self.collection.query(
                    query_texts=[question],
                    n_results=min(10, self.total_documents)  # Limit to 10 most relevant docs
                )
                
                if not results or not results['documents'] or not results['documents'][0]:
                    st.warning("No specific documents found relevant to your query. Using overall dataset analysis.")
                    relevant_docs = []
                else:
                    relevant_docs = results['documents'][0][:5]  # Use only top 5 docs to keep prompt size manageable
                
                # Prepare comprehensive prompt
                # Use the converter here to ensure all data is serializable
                prompt = "Provide a detailed analysis based on this dataset:\n\nOverall Metrics:\n" + json.dumps(self.convert_to_serializable(analysis['overall_metrics']), indent=2) + "\n\nClient-Specific Analysis:\n" + json.dumps(self.convert_to_serializable(analysis['client_analysis']), indent=2) + "\n\nRecent Trends:\n" + json.dumps(self.convert_to_serializable(analysis['recent_trends']), indent=2) + "\n\nShipping Company Performance:\n" + json.dumps(self.convert_to_serializable(analysis['shipping_performance']), indent=2)
                
                # Add relevant documents if available
                if relevant_docs:
                    docs_text = "\nRelevant Employee Records:\n"
                    for i, doc in enumerate(relevant_docs):
                        docs_text += f"Record {i+1}:\n{doc}\n\n"
                    prompt += docs_text
                
                prompt += "\nQuestion: " + question + "\n\nProvide a comprehensive answer that:\n1. Addresses the specific question with precise data points\n2. Includes relevant metrics and trends with numbers\n3. Provides specific examples from the data\n4. Considers the entire dataset context\n5. Highlights key patterns and areas for improvement\n6. Identifies critical issues and their potential impact"

                # Get analysis with recommendations
                answer = self.query_lmstudio(prompt, max_tokens=2000)
                
                if answer:
                    # Generate structured recommendations
                    recommendations = self.generate_structured_recommendations(answer)
                    
                    # Generate additional insights
                    summary_prompt = "Based on this analysis, provide a concise, data-driven summary of the key findings, focusing on the most important metrics and patterns: " + answer
                    
                    prediction_prompt = "Based on this analysis, provide detailed, actionable recommendations:\n1. For each major area (shipping, onboarding, IT asset management), provide specific, implementable actions\n2. Include expected impact of each recommendation with estimated improvement percentages\n3. Highlight priority areas that need immediate attention with specific metrics to track\n4. Suggest preventive measures for recurring issues\n5. Recommend process improvements with specific implementation steps\n\nAnalysis: " + answer
                    
                    summary = self.query_lmstudio(summary_prompt, max_tokens=800)
                    predictions = self.query_lmstudio(prediction_prompt, max_tokens=1500)
                    
                    return {
                        "answer": answer,
                        "summary": summary if summary else "Summary not available",
                        "predictions": predictions if predictions else "Recommendations not available",
                        "structured_recommendations": recommendations
                    }
                else:
                    raise Exception("Failed to generate analysis")
                    
        except Exception as e:
            st.error(f"❌ Error in analysis process: {str(e)}")
            return {
                "error": str(e),
                "answer": None,
                "summary": None,
                "predictions": None,
                "structured_recommendations": []
            }

    def initialize_system(self):
        """Initialize the system with pre-populated ChromaDB"""
        try:
            with st.spinner("🚀 Initializing system..."):
                self.setup_chromadb()
            
            st.success("✅ System initialization complete!")
            st.markdown("### 📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(self.dataframe))
            with col2:
                st.metric("Unique Clients", self.dataframe['ClientName'].nunique())
            with col3:
                completed_setups = self.dataframe['RemoteSetupComplete'].fillna(False).astype(bool).sum()
                st.metric("Completed Setups", completed_setups)
            with col4:
                completion_rate = (completed_setups / len(self.dataframe)) * 100
                st.metric("Setup Completion Rate", f"{completion_rate:.1f}%")
            
            return True
        except Exception as e:
            st.error(f"❌ Error during system initialization: {str(e)}")
            return False

def main():
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'query_system' not in st.session_state:
        st.session_state.query_system = None
    
    st.title("Employee Data Analysis System 👥")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        lmstudio_url = st.text_input(
            "LM Studio URL",
            value="http://localhost:1234",
            help="Enter the URL where LM Studio is running"
        )
        
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System", use_container_width=True):
                system = EmployeeDataQuerySystem(lmstudio_url)
                if system.initialize_system():
                    st.session_state.system_initialized = True
                    st.session_state.query_system = system
                    st.rerun()
    
    if st.session_state.system_initialized:
        st.header("❓ Ask Questions")
        
        with st.expander("📝 Example Questions"):
            st.markdown("""
            - What is the average delivery time for IT assets? Break down by client.
            - Show me a detailed analysis of all in-transit assets and their current status.
            - Which clients have the highest and lowest setup completion rates?
            - Analyze the onboarding timeline trends across all clients.
            - What are the common shipping delays and their patterns?
            - Compare remote setup completion rates between different clients.
            - Give me a full analysis of asset delivery performance by shipping company.
            - What recommendations can you provide to improve our overall onboarding process?
            - Which client is experiencing the most delays in IT asset delivery?
            - How can we improve our shipping process for better delivery times?
            """)
        
        st.markdown("""
        💡 **Tip**: Ask detailed questions to get comprehensive analysis of the entire dataset.
        The system will analyze all records and provide detailed insights with specific numbers and trends.
        """)
        
        question = st.text_area("Enter your question:", height=100,
                              help="Ask about any aspect of the employee dataset")
        
        if st.button("🔍 Get Answer", type="primary"):
            if question:
                with st.spinner("🧠 Analyzing entire dataset..."):
                    response = st.session_state.query_system.query(question)
                
                if response.get("error"):
                    st.error(response["error"])
                else:
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Detailed Analysis", "📊 Key Findings", "🔮 Recommendations", "📈 Structured Insights"])
                    
                    with tab1:
                        st.markdown("### Comprehensive Analysis")
                        st.write(response["answer"])
                    
                    with tab2:
                        st.markdown("### Summary of Key Findings")
                        st.write(response["summary"])
                    
                    with tab3:
                        st.markdown("### Detailed Recommendations")
                        st.write(response["predictions"])
                        
                    with tab4:
                        st.markdown("### Structured Recommendations")
                        recommendations = response.get("structured_recommendations", [])
                        
                        if recommendations:
                            for i, rec in enumerate(recommendations):
                                with st.container(border=True):
                                    cols = st.columns([3, 1])
                                    with cols[0]:
                                        st.markdown(f"### {rec.get('category', 'General Recommendation')}")
                                        st.markdown(f"**{rec.get('recommendation', '')}**")
                                        st.markdown(f"Impact Area: {rec.get('impact_area', 'Various')}")
                                    
                                    with cols[1]:
                                        priority = rec.get('priority', 3)
                                        difficulty = rec.get('implementation_difficulty', 3)
                                        improvement = rec.get('estimated_improvement', 0)
                                        
                                        st.metric("Priority", f"{priority}/5")
                                        st.metric("Difficulty", f"{difficulty}/5")
                                        st.metric("Est. Improvement", f"{improvement}%")
                        else:
                            st.info("No structured recommendations available for this query.")
                    
                    st.markdown("---")
                    st.info("💡 Analysis based on complete dataset processing")
            else:
                st.warning("⚠️ Please enter a question.")
    else:
        st.info("👈 Click the 'Initialize System' button in the sidebar to connect to the ChromaDB database.")
        
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            ### Employee Data Analysis System
            
            This application provides comprehensive analysis of employee onboarding data, with focus on:
            
            - **IT Asset Management**: Track asset shipping, delivery, and setup status
            - **Client-Based Analysis**: Compare metrics across different clients
            - **Recommendation Engine**: Get actionable insights to improve processes
            - **ChromaDB Integration**: Direct access to pre-populated ChromaDB data
            
            The system uses a local LM Studio instance for generating insights and recommendations.
            
            **Note**: This version assumes ChromaDB has already been populated with employee data
            using the separate ingestion script.
            """)

if __name__ == "__main__":
    main()