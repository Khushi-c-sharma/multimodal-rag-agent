"""
Streamlit App for Multimodal RAG System
Features:
- Interactive query interface
- Real-time evaluation metrics
- Latency tracking
- Visual results display
"""

import streamlit as st
import time
import asyncio
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Import from your modules
from dual_qa_setup import setup_system, ParallelRAGClipOnly
from evaluation_metrics import (
    calculate_retrieval_metrics,
    calculate_diversity_score,
    calculate_mrr,
    calculate_ndcg
)


# Page config
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .image-result {
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system(text_path: str, images_path: str, config: Dict):
    """Initialize RAG system (cached)."""
    with st.spinner("🔄 Loading FAISS indexes and models..."):
        agent = setup_system(
            text_path,
            images_path,
            clip_model=config['clip_model'],
            top_k=config['top_k'],
            retrieval_k=config['retrieval_k'],
            fetch_k=config['fetch_k'],
            lambda_mult=config['lambda_mult'],
            rerank_lambda=config['rerank_lambda']
        )
    return agent


def initialize_session_state():
    """Initialize session state variables."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0


def query_with_metrics(agent: ParallelRAGClipOnly, query: str) -> Dict[str, Any]:
    """Execute query and collect metrics."""
    metrics = {
        'query': query,
        'timestamp': datetime.now(),
    }
    
    # Retrieval timing
    start_time = time.time()
    result = agent.ask(query)
    total_time = time.time() - start_time
    
    metrics['total_latency'] = total_time
    metrics['num_results'] = len(result['top_items'])
    
    # Calculate retrieval metrics
    if result['top_items']:
        metrics['avg_score'] = sum(item['score'] for item in result['top_items']) / len(result['top_items'])
        metrics['max_score'] = max(item['score'] for item in result['top_items'])
        metrics['min_score'] = min(item['score'] for item in result['top_items'])
        metrics['diversity'] = calculate_diversity_score(result['top_items'])
        
        # Type distribution
        text_count = sum(1 for item in result['top_items'] if item['type'] == 'text')
        img_count = sum(1 for item in result['top_items'] if item['type'] == 'image')
        metrics['text_count'] = text_count
        metrics['image_count'] = img_count
    
    return {**result, **metrics}


def display_metrics_dashboard(metrics: Dict[str, Any]):
    """Display comprehensive metrics dashboard."""
    st.markdown("### 📊 Retrieval Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⏱️ Total Latency",
            f"{metrics['total_latency']:.2f}s",
            help="End-to-end query processing time"
        )
    
    with col2:
        st.metric(
            "📄 Results",
            metrics['num_results'],
            help="Total number of retrieved items"
        )
    
    with col3:
        st.metric(
            "🎯 Avg Score",
            f"{metrics.get('avg_score', 0):.3f}",
            help="Average relevance score"
        )
    
    with col4:
        st.metric(
            "🌈 Diversity",
            f"{metrics.get('diversity', 0):.3f}",
            help="Result diversity score (higher = more diverse)"
        )
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Score Distribution")
        fig_scores = go.Figure()
        fig_scores.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics.get('max_score', 0),
            title="Max Score",
            delta={'reference': metrics.get('avg_score', 0)},
            domain={'x': [0, 0.5], 'y': [0.5, 1]}
        ))
        fig_scores.add_trace(go.Indicator(
            mode="number+delta",
            value=metrics.get('min_score', 0),
            title="Min Score",
            delta={'reference': metrics.get('avg_score', 0)},
            domain={'x': [0.5, 1], 'y': [0.5, 1]}
        ))
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        st.markdown("#### Result Type Distribution")
        if metrics.get('text_count', 0) + metrics.get('image_count', 0) > 0:
            fig_pie = px.pie(
                values=[metrics.get('text_count', 0), metrics.get('image_count', 0)],
                names=['Text', 'Image'],
                color_discrete_sequence=['#636EFA', '#EF553B']
            )
            st.plotly_chart(fig_pie, use_container_width=True)


def display_results(result: Dict[str, Any]):
    """Display query results."""
    st.markdown("### 💬 Answer")
    st.info(result['answer'])
    
    st.markdown("### 🔍 Retrieved Items")
    
    for i, item in enumerate(result['top_items'], 1):
        with st.expander(f"**{i}. [{item['type'].upper()}]** Score: {item['score']:.3f}"):
            if item['type'] == 'image':
                col1, col2 = st.columns([1, 2])
                with col1:
                    image_path = item.get('image_path', '')
                    if image_path:
                        try:
                            st.image(image_path, use_container_width=True)
                        except:
                            st.warning(f"Could not load image: {image_path}")
                with col2:
                    st.markdown(f"**Caption:** {item['content']}")
                    st.markdown(f"**Path:** `{image_path}`")
            else:
                st.markdown(item['content'])
            
            # Metadata
            if item.get('metadata'):
                with st.expander("View Metadata"):
                    st.json(item['metadata'])


def display_history_analytics():
    """Display analytics from query history."""
    if not st.session_state.metrics_history:
        st.info("No query history yet. Run some queries to see analytics!")
        return
    
    st.markdown("### 📈 Performance Analytics")
    
    df = pd.DataFrame(st.session_state.metrics_history)
    
    # Latency over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Latency Trends")
        fig_latency = px.line(
            df,
            x=df.index,
            y='total_latency',
            title='Query Latency Over Time',
            labels={'x': 'Query Number', 'total_latency': 'Latency (s)'}
        )
        fig_latency.add_hline(
            y=df['total_latency'].mean(),
            line_dash="dash",
            annotation_text=f"Avg: {df['total_latency'].mean():.2f}s"
        )
        st.plotly_chart(fig_latency, use_container_width=True)
    
    with col2:
        st.markdown("#### Score Distribution")
        fig_scores = px.box(
            df,
            y=['avg_score', 'max_score', 'min_score'],
            title='Relevance Score Distribution'
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Latency",
            f"{df['total_latency'].mean():.2f}s",
            f"±{df['total_latency'].std():.2f}s"
        )
    
    with col2:
        st.metric(
            "Avg Diversity",
            f"{df['diversity'].mean():.3f}",
            f"±{df['diversity'].std():.3f}"
        )
    
    with col3:
        st.metric(
            "Total Queries",
            len(df)
        )
    
    # Recent queries table
    st.markdown("#### Recent Queries")
    recent_df = df[['query', 'total_latency', 'num_results', 'avg_score', 'diversity']].tail(10)
    st.dataframe(
        recent_df.style.format({
            'total_latency': '{:.2f}s',
            'avg_score': '{:.3f}',
            'diversity': '{:.3f}'
        }),
        width='stretch'
    )


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Multimodal RAG System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Paths
    st.sidebar.markdown("### 📁 Index Paths")
    text_path = st.sidebar.text_input(
        "Text Tables Path",
        value="./faiss_indexes/text_tables"
    )
    images_path = st.sidebar.text_input(
        "Images Path",
        value="./faiss_indexes/images"
    )
    
    # Model config
    st.sidebar.markdown("### 🤖 Model Settings")
    clip_model = st.sidebar.selectbox(
        "CLIP Model",
        ["clip-ViT-B-32", "clip-ViT-L-14"],
        index=0
    )
    
    # Retrieval config
    st.sidebar.markdown("### 🔧 Retrieval Settings")
    top_k = st.sidebar.slider("Top K Results", 1, 20, 6)
    retrieval_k = st.sidebar.slider("Retrieval K", 5, 50, 20)
    fetch_k = st.sidebar.slider("Fetch K (MMR pool)", 10, 100, 50)
    
    # Diversity settings
    st.sidebar.markdown("### 🌈 Diversity Control")
    lambda_mult = st.sidebar.slider(
        "Retrieval Diversity (λ)",
        0.0, 1.0, 0.5,
        help="0=max diversity, 1=max relevance"
    )
    rerank_lambda = st.sidebar.slider(
        "Reranking Diversity (λ)",
        0.0, 1.0, 0.7,
        help="0=max diversity, 1=max relevance"
    )
    
    config = {
        'clip_model': clip_model,
        'top_k': top_k,
        'retrieval_k': retrieval_k,
        'fetch_k': fetch_k,
        'lambda_mult': lambda_mult,
        'rerank_lambda': rerank_lambda
    }
    
    # Initialize system
    try:
        agent = initialize_system(text_path, images_path, config)
        st.sidebar.success("✅ System initialized!")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Query", "📊 Metrics", "📈 Analytics"])
    
    with tab1:
        st.markdown("### Enter your query")
        
        # Sample queries
        sample_queries = [
            "Explain Qatar's GDP trend and include relevant images.",
            "What are the major economic indicators?",
            "Show me information about renewable energy investments",
            "Compare inflation rates across different countries"
        ]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Query",
                placeholder="Enter your question here...",
                label_visibility="collapsed"
            )
        with col2:
            use_sample = st.selectbox(
                "Or use sample",
                ["Custom"] + sample_queries,
                label_visibility="collapsed"
            )
        
        if use_sample != "Custom":
            query = use_sample
        
        if st.button("🚀 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("🔄 Processing your query..."):
                    result = query_with_metrics(agent, query)
                    
                    # Store in history
                    st.session_state.query_history.append(result)
                    st.session_state.metrics_history.append({
                        'query': result['query'],
                        'total_latency': result['total_latency'],
                        'num_results': result['num_results'],
                        'avg_score': result.get('avg_score', 0),
                        'max_score': result.get('max_score', 0),
                        'min_score': result.get('min_score', 0),
                        'diversity': result.get('diversity', 0),
                        'text_count': result.get('text_count', 0),
                        'image_count': result.get('image_count', 0)
                    })
                    st.session_state.total_queries += 1
                
                # Display results
                display_results(result)
            else:
                st.warning("⚠️ Please enter a query")
    
    with tab2:
        if st.session_state.query_history:
            latest_result = st.session_state.query_history[-1]
            
            st.markdown(f"**Latest Query:** {latest_result['query']}")
            st.markdown("---")
            
            display_metrics_dashboard(latest_result)
            
            # Score distribution chart
            if latest_result['top_items']:
                st.markdown("### 📊 Item Score Distribution")
                items_df = pd.DataFrame([
                    {
                        'Item': f"{i+1}. {item['type']}",
                        'Score': item['score'],
                        'Type': item['type']
                    }
                    for i, item in enumerate(latest_result['top_items'])
                ])
                
                fig = px.bar(
                    items_df,
                    x='Item',
                    y='Score',
                    color='Type',
                    title='Relevance Scores by Retrieved Item'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 Run a query to see detailed metrics")
    
    with tab3:
        display_history_analytics()
        
        # Export options
        if st.session_state.metrics_history:
            st.markdown("### 💾 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Metrics CSV"):
                    df = pd.DataFrame(st.session_state.metrics_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download CSV",
                        csv,
                        "metrics_history.csv",
                        "text/csv"
                    )
            
            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.query_history = []
                    st.session_state.metrics_history = []
                    st.session_state.total_queries = 0
                    st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total Queries:** {st.session_state.total_queries}")
    if st.session_state.metrics_history:
        avg_latency = sum(m['total_latency'] for m in st.session_state.metrics_history) / len(st.session_state.metrics_history)
        st.sidebar.markdown(f"**Avg Latency:** {avg_latency:.2f}s")


if __name__ == "__main__":
    main()