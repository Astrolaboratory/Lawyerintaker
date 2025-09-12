import streamlit as st
import openai
import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Legal Intake AI Classifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class ClassificationResult:
    """Data class for classification results"""
    case_type: str
    urgency_level: str
    status_recommendation: str
    reasoning: str
    confidence_score: float
    key_factors: List[str]
    statute_of_limitations_concern: bool
    estimated_case_value: str
    timestamp: str = ""

class StreamlitLegalClassifier:
    """Streamlit-optimized legal intake classifier"""
    
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.client = openai.OpenAI(api_key=api_key)
        self.case_types = {
            "Motor Vehicle Accident": ["car", "vehicle", "accident", "collision", "rear-end", "traffic"],
            "Medical Malpractice": ["doctor", "hospital", "medical", "surgery", "diagnosis", "treatment"],
            "Workers' Compensation": ["work", "job", "workplace", "construction", "osha", "injury"],
            "Premises Liability": ["slip", "fall", "store", "property", "premises", "hazard"],
            "Product Liability": ["product", "defective", "recall", "manufacturer", "malfunction"],
            "Employment Law": ["fired", "discrimination", "harassment", "wrongful", "termination"],
            "Family Law": ["divorce", "custody", "child support", "marriage", "domestic"],
            "Personal Injury": ["injury", "hurt", "pain", "negligence", "damages"],
            "Contract Dispute": ["contract", "agreement", "breach", "business", "payment"],
            "General Inquiry": ["question", "consultation", "advice", "help"]
        }
        
    def classify_intake(self, intake_text: str) -> ClassificationResult:
        """Classify legal intake with error handling"""
        try:
            prompt = self._build_classification_prompt(intake_text)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert legal intake specialist. Provide accurate case classifications in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return ClassificationResult(
                case_type=result.get('case_type', 'General Inquiry'),
                urgency_level=result.get('urgency_level', 'Medium'),
                status_recommendation=result.get('status_recommendation', 'Needs More Info'),
                reasoning=result.get('reasoning', 'Analysis completed.'),
                confidence_score=float(result.get('confidence_score', 0.5)),
                key_factors=result.get('key_factors', []),
                statute_of_limitations_concern=result.get('statute_of_limitations_concern', False),
                estimated_case_value=result.get('estimated_case_value', 'Unknown'),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
            return self._get_fallback_classification(intake_text)
    
    def _build_classification_prompt(self, intake_text: str) -> str:
        """Build classification prompt"""
        return f"""
        Analyze this legal intake and classify it:

        CASE TYPES: Motor Vehicle Accident, Medical Malpractice, Workers' Compensation, 
        Premises Liability, Product Liability, Employment Law, Family Law, Personal Injury, 
        Contract Dispute, General Inquiry

        URGENCY: High (urgent/severe), Medium (moderate), Low (minor/routine)
        
        STATUS: Qualified Lead (strong case), Needs More Info (potential), Not a Case (no merit)

        CLIENT INTAKE:
        {intake_text}

        Respond in JSON format:
        {{
            "case_type": "specific case type",
            "urgency_level": "High/Medium/Low",
            "status_recommendation": "status",
            "reasoning": "detailed analysis",
            "confidence_score": 0.85,
            "key_factors": ["factor1", "factor2"],
            "statute_of_limitations_concern": true/false,
            "estimated_case_value": "Low/Medium/High/Unknown"
        }}
        """
    
    def _get_fallback_classification(self, intake_text: str) -> ClassificationResult:
        """Fallback classification using keyword matching"""
        text_lower = intake_text.lower()
        
        # Simple keyword matching
        best_match = "General Inquiry"
        best_score = 0
        
        for case_type, keywords in self.case_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_match = case_type
        
        return ClassificationResult(
            case_type=best_match,
            urgency_level="Medium",
            status_recommendation="Needs More Info",
            reasoning="Fallback classification used due to API error. Manual review recommended.",
            confidence_score=0.4,
            key_factors=["API Error", "Keyword-based classification"],
            statute_of_limitations_concern=False,
            estimated_case_value="Unknown",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

def get_sample_cases():
    """Sample cases for testing"""
    return {
        "Car Accident": """I was rear-ended at a red light last week. The other driver was texting and didn't brake at all. 
        I have severe whiplash and my car is totaled. I've been to the ER and am starting physical therapy. 
        The police report shows the other driver was at fault. My medical bills are already over $8,000.""",
        
        "Medical Malpractice": """My mother went to the hospital with severe chest pain. The doctor said it was just anxiety 
        and sent her home. Three hours later she had a massive heart attack. If they had done proper tests, 
        they would have seen the blockage. She survived but has permanent heart damage now.""",
        
        "Workplace Injury": """I fell from scaffolding at my construction job. The safety equipment wasn't properly maintained 
        and OSHA cited my employer for violations. I broke my leg in three places and can't work. 
        Workers' comp is denying my claim even though it was clearly a workplace accident.""",
        
        "Slip and Fall": """I slipped on a wet floor at the grocery store. There was no warning sign and the spill 
        had been there for over an hour according to witnesses. I broke my hip and needed surgery. 
        I'm 72 years old and this has really impacted my mobility and independence."""
    }

def create_classification_chart(result: ClassificationResult):
    """Create visualization of classification results"""
    
    # Confidence gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result.confidence_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    
    return fig_gauge

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("⚖️ Legal Intake AI Classifier")
    st.markdown("### Intelligent Case Classification and Triage System")
    
    # Sidebar for API key and settings
    st.sidebar.header("🔧 Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Enter your OpenAI API key to enable AI classification"
    )
    
    # Initialize classifier
    classifier = None
    if api_key:
        try:
            classifier = StreamlitLegalClassifier(api_key)
            st.sidebar.success("✅ AI Classifier Ready")
        except Exception as e:
            st.sidebar.error(f"❌ Error initializing classifier: {str(e)}")
    else:
        st.sidebar.warning("⚠️ API key required for AI classification")
    
    # Mode selection
    mode = st.sidebar.radio(
        "📋 Select Mode",
        ["Single Case Analysis", "Sample Cases", "Batch Processing", "Analytics Dashboard"]
    )
    
    # Initialize session state
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
    
    if mode == "Single Case Analysis":
        st.header("📝 Single Case Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            intake_text = st.text_area(
                "Enter Case Intake Details:",
                height=200,
                placeholder="Describe the legal issue, including relevant facts, damages, timeline, and any evidence..."
            )
            
            # Analysis button
            if st.button("🔍 Analyze Case", type="primary"):
                if not intake_text.strip():
                    st.error("Please enter case details to analyze.")
                elif len(intake_text.strip()) < 20:
                    st.error("Please provide more detailed information (at least 20 characters).")
                elif not classifier:
                    st.error("Please enter your OpenAI API key in the sidebar.")
                else:
                    with st.spinner("🤖 Analyzing case with AI..."):
                        result = classifier.classify_intake(intake_text)
                        
                        # Store in session state
                        st.session_state.results_history.append({
                            'intake': intake_text,
                            'result': result,
                            'timestamp': result.timestamp
                        })
                        
                        # Display results
                        display_classification_result(result, intake_text)
        
        with col2:
            st.subheader("📊 Quick Stats")
            if st.session_state.results_history:
                total_cases = len(st.session_state.results_history)
                avg_confidence = sum(r['result'].confidence_score for r in st.session_state.results_history) / total_cases
                
                st.metric("Total Cases Analyzed", total_cases)
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
                
                # Case type distribution
                case_types = [r['result'].case_type for r in st.session_state.results_history]
                case_type_counts = pd.Series(case_types).value_counts()
                
                if len(case_type_counts) > 0:
                    fig = px.pie(
                        values=case_type_counts.values,
                        names=case_type_counts.index,
                        title="Case Type Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "Sample Cases":
        st.header("📋 Sample Cases")
        
        sample_cases = get_sample_cases()
        
        selected_case = st.selectbox(
            "Choose a sample case:",
            list(sample_cases.keys())
        )
        
        if selected_case:
            st.subheader(f"Sample: {selected_case}")
            st.text_area("Case Details:", sample_cases[selected_case], height=150, disabled=True)
            
            if st.button("🔍 Analyze Sample Case", type="primary"):
                if classifier:
                    with st.spinner("🤖 Analyzing sample case..."):
                        result = classifier.classify_intake(sample_cases[selected_case])
                        display_classification_result(result, sample_cases[selected_case])
                        
                        # Add to history
                        st.session_state.results_history.append({
                            'intake': sample_cases[selected_case],
                            'result': result,
                            'timestamp': result.timestamp
                        })
                else:
                    st.error("Please enter your OpenAI API key in the sidebar.")
    
    elif mode == "Batch Processing":
        st.header("📦 Batch Processing")
        
        st.markdown("Upload multiple cases for batch analysis:")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with cases",
            type=['csv'],
            help="CSV should have a column named 'intake_text' with case descriptions"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} cases from file")
                
                if 'intake_text' in df.columns:
                    st.dataframe(df.head())
                    
                    if st.button("🚀 Process All Cases", type="primary"):
                        if classifier:
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, row in df.iterrows():
                                intake_text = row['intake_text']
                                result = classifier.classify_intake(intake_text)
                                results.append(result)
                                progress_bar.progress((i + 1) / len(df))
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame([asdict(r) for r in results])
                            
                            st.success(f"✅ Processed {len(results)} cases successfully!")
                            st.dataframe(results_df)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("Please enter your OpenAI API key in the sidebar.")
                else:
                    st.error("CSV file must contain an 'intake_text' column")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif mode == "Analytics Dashboard":
        st.header("📊 Analytics Dashboard")
        
        if st.session_state.results_history:
            results_df = pd.DataFrame([
                {
                    **asdict(r['result']),
                    'intake_preview': r['intake'][:100] + "..." if len(r['intake']) > 100 else r['intake']
                }
                for r in st.session_state.results_history
            ])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cases", len(results_df))
            with col2:
                qualified_leads = len(results_df[results_df['status_recommendation'] == 'Qualified Lead'])
                st.metric("Qualified Leads", qualified_leads)
            with col3:
                avg_confidence = results_df['confidence_score'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            with col4:
                high_urgency = len(results_df[results_df['urgency_level'] == 'High'])
                st.metric("High Urgency", high_urgency)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Case type distribution
                case_type_counts = results_df['case_type'].value_counts()
                fig1 = px.bar(
                    x=case_type_counts.values,
                    y=case_type_counts.index,
                    orientation='h',
                    title="Cases by Type",
                    labels={'x': 'Count', 'y': 'Case Type'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Urgency vs Status
                urgency_status = results_df.groupby(['urgency_level', 'status_recommendation']).size().reset_index(name='count')
                fig2 = px.bar(
                    urgency_status,
                    x='urgency_level',
                    y='count',
                    color='status_recommendation',
                    title="Urgency vs Status Distribution",
                    barmode='stack'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            st.dataframe(
                results_df[['case_type', 'urgency_level', 'status_recommendation', 'confidence_score', 'estimated_case_value']],
                use_container_width=True
            )
            
            # Export options
            st.subheader("📥 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            with col2:
                json_data = results_df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        else:
            st.info("📈 No data available yet. Analyze some cases first!")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 Privacy & Security")
    st.sidebar.info("Your API key and case data are not stored permanently. Data is only kept in your session.")
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.results_history = []
        st.sidebar.success("History cleared!")

def display_classification_result(result: ClassificationResult, intake_text: str):
    """Display classification results in a nice format"""
    
    st.success("✅ Analysis Complete!")
    
    # Main result cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Case Type",
            result.case_type,
            help="The most likely legal case category"
        )
    
    with col2:
        urgency_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        st.metric(
            "Urgency Level",
            f"{urgency_color.get(result.urgency_level, '⚪')} {result.urgency_level}",
            help="Priority level for case handling"
        )
    
    with col3:
        status_color = {
            "Qualified Lead": "🟢",
            "Needs More Info": "🟡", 
            "Not a Case": "🔴"
        }
        st.metric(
            "Status",
            f"{status_color.get(result.status_recommendation, '⚪')} {result.status_recommendation}",
            help="Recommended next action"
        )
    
    # Detailed information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🧠 AI Analysis")
        st.write(result.reasoning)
        
        if result.key_factors:
            st.subheader("🔑 Key Factors")
            for factor in result.key_factors:
                st.write(f"• {factor}")
    
    with col2:
        # Confidence gauge
        fig_gauge = create_classification_chart(result)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Additional metrics
        st.metric("Case Value Estimate", result.estimated_case_value)
        st.metric(
            "Statute Concern",
            "Yes" if result.statute_of_limitations_concern else "No",
            help="Potential statute of limitations issues"
        )
    
    # Case details in expander
    with st.expander("📄 View Original Intake"):
        st.text_area("Original case details:", intake_text, height=150, disabled=True)

if __name__ == "__main__":
    main()