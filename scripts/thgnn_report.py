#!/usr/bin/env python3
"""
THGNN Ultimate Report Generator - Production Grade
The definitive reporting script for professional threat analysis reports.

Features:
- Executive summary reports for leadership
- Technical analysis reports for security teams
- Interactive visualizations and charts
- MITRE ATT&CK framework integration
- Professional PDF and HTML output
- Customizable templates and branding

Output Formats:
- Executive PDF: High-level risk assessment
- Technical HTML: Detailed analysis with interactive charts
- JSON: Machine-readable data export
- CSV: Tabular data for further analysis

Author: THGNN Expert
Version: 2.0 - ULTIMATE PRODUCTION
Date: February 2026
"""

import os
import sys
import json
import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Template, Environment, FileSystemLoader
import pdfkit
from weasyprint import HTML, CSS

# Suppress warnings for production
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/thgnn_report.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Report generation configuration"""
    title: str = "THGNN Threat Analysis Report"
    organization: str = "Security Operations Center"
    classification: str = "CONFIDENTIAL"
    include_executive_summary: bool = True
    include_technical_details: bool = True
    include_visualizations: bool = True
    include_mitre_mapping: bool = True
    include_recommendations: bool = True
    output_formats: List[str] = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['html', 'pdf', 'json']

class UltimateTHGNNReportGenerator:
    """
    Ultimate THGNN report generator for professional threat analysis.
    
    Generates comprehensive reports combining:
    - Executive summaries for leadership
    - Technical analysis for security teams
    - Interactive visualizations
    - MITRE ATT&CK framework mapping
    - Actionable recommendations
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize the ultimate report generator"""
        logger.info("📊 Initializing Ultimate THGNN Report Generator")
        
        self.config = config or ReportConfig()
        self.output_dir = project_root / "outputs" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Template directory
        self.template_dir = project_root / "templates" / "reports"
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        
        # Create default templates if they don't exist
        self._create_default_templates()
        
        # MITRE ATT&CK technique descriptions
        self.mitre_descriptions = self._load_mitre_descriptions()
        
        logger.info("✅ Report generator initialized successfully")
    
    def _create_default_templates(self) -> None:
        """Create default report templates"""
        # Executive summary template
        executive_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Executive Summary</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }
        .header { text-align: center; border-bottom: 3px solid #2c5aa0; padding-bottom: 20px; margin-bottom: 30px; }
        .classification { color: #d32f2f; font-weight: bold; font-size: 18px; }
        .summary-box { background: #f5f5f5; padding: 20px; border-left: 5px solid #2c5aa0; margin: 20px 0; }
        .risk-critical { color: #d32f2f; font-weight: bold; }
        .risk-high { color: #f57c00; font-weight: bold; }
        .risk-medium { color: #1976d2; font-weight: bold; }
        .risk-low { color: #388e3c; font-weight: bold; }
        .metric { display: inline-block; margin: 10px 20px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c5aa0; }
        .metric-label { font-size: 12px; color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #2c5aa0; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <h2>Executive Summary</h2>
        <p class="classification">{{ classification }}</p>
        <p>{{ organization }} | {{ timestamp }}</p>
    </div>
    
    <div class="summary-box">
        <h3>Risk Assessment</h3>
        <p>Overall Risk Level: <span class="risk-{{ overall_risk.lower() }}">{{ overall_risk }}</span></p>
        <p>{{ risk_description }}</p>
    </div>
    
    <div style="text-align: center; margin: 30px 0;">
        <div class="metric">
            <div class="metric-value">{{ total_threats }}</div>
            <div class="metric-label">Total Threats</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ critical_threats }}</div>
            <div class="metric-label">Critical</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ high_threats }}</div>
            <div class="metric-label">High</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ entities_at_risk }}</div>
            <div class="metric-label">Entities at Risk</div>
        </div>
    </div>
    
    {% if top_threats %}
    <h3>Top Threats Detected</h3>
    <table>
        <tr>
            <th>Rank</th>
            <th>Source</th>
            <th>Target</th>
            <th>Attack Type</th>
            <th>Risk Score</th>
            <th>MITRE Technique</th>
        </tr>
        {% for threat in top_threats[:10] %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ threat.source_entity }}</td>
            <td>{{ threat.target_entity }}</td>
            <td>{{ threat.edge_type }}</td>
            <td>{{ "%.3f"|format(threat.risk_score) }}</td>
            <td>{{ threat.mitre_techniques[:2]|join(', ') if threat.mitre_techniques else 'N/A' }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
    
    <h3>Key Recommendations</h3>
    <ul>
        {% for recommendation in recommendations %}
        <li>{{ recommendation }}</li>
        {% endfor %}
    </ul>
    
    <div style="margin-top: 50px; font-size: 12px; color: #666;">
        <p>This report was generated by the THGNN (Temporal Heterogeneous Graph Neural Network) system.</p>
        <p>Report generated on {{ timestamp }} | Processing time: {{ processing_time }} seconds</p>
    </div>
</body>
</html>
        """
        
        exec_template_path = self.template_dir / "executive_summary.html"
        if not exec_template_path.exists():
            with open(exec_template_path, 'w', encoding='utf-8') as f:
                f.write(executive_template.strip())
        
        # Technical report template
        technical_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Technical Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; color: #333; line-height: 1.6; }
        .header { text-align: center; border-bottom: 3px solid #2c5aa0; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin: 30px 0; }
        .chart-container { margin: 20px 0; height: 400px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #2c5aa0; color: white; }
        .code { background: #f5f5f5; padding: 10px; border-left: 3px solid #2c5aa0; font-family: monospace; }
        .mitre-technique { background: #e3f2fd; padding: 5px; border-radius: 3px; margin: 2px; display: inline-block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <h2>Technical Analysis Report</h2>
        <p>{{ organization }} | {{ timestamp }}</p>
    </div>
    
    <div class="section">
        <h3>Analysis Overview</h3>
        <p><strong>Total Events Processed:</strong> {{ total_events:,d }}</p>
        <p><strong>Total Entities:</strong> {{ total_entities:,d }}</p>
        <p><strong>Processing Time:</strong> {{ processing_time }} seconds</p>
        <p><strong>Model Performance:</strong> AUC-ROC {{ model_auc }}</p>
    </div>
    
    <div class="section">
        <h3>Threat Distribution</h3>
        <div id="threat-distribution-chart" class="chart-container"></div>
    </div>
    
    <div class="section">
        <h3>MITRE ATT&CK Techniques Detected</h3>
        {% for technique in mitre_techniques %}
        <span class="mitre-technique">{{ technique }}</span>
        {% endfor %}
    </div>
    
    <div class="section">
        <h3>Attack Chains</h3>
        {% for chain in attack_chains %}
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <h4>Chain {{ loop.index }}: {{ chain.attack_pattern or 'Unknown Pattern' }}</h4>
            <p><strong>Severity:</strong> {{ chain.severity }}</p>
            <p><strong>Confidence:</strong> {{ "%.2f"|format(chain.confidence) }}</p>
            <p><strong>Steps:</strong> {{ chain.total_steps }}</p>
            <p><strong>Duration:</strong> {{ "%.1f"|format(chain.duration_seconds) }} seconds</p>
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h3>Detailed Threat Analysis</h3>
        <table>
            <tr>
                <th>ID</th>
                <th>Source Entity</th>
                <th>Target Entity</th>
                <th>Attack Type</th>
                <th>Risk Score</th>
                <th>Confidence</th>
                <th>MITRE Techniques</th>
            </tr>
            {% for threat in detailed_threats %}
            <tr>
                <td>{{ threat.prediction_id[:8] }}</td>
                <td>{{ threat.source_entity }}</td>
                <td>{{ threat.target_entity }}</td>
                <td>{{ threat.edge_type }}</td>
                <td>{{ "%.4f"|format(threat.risk_score) }}</td>
                <td>{{ "%.4f"|format(threat.confidence) }}</td>
                <td>{{ threat.mitre_techniques|join(', ') if threat.mitre_techniques else 'N/A' }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <script>
        // Threat distribution chart
        var threatData = [{
            x: ['Critical', 'High', 'Medium', 'Low'],
            y: [{{ critical_threats }}, {{ high_threats }}, {{ medium_threats }}, {{ low_threats }}],
            type: 'bar',
            marker: {
                color: ['#d32f2f', '#f57c00', '#1976d2', '#388e3c']
            }
        }];
        
        var layout = {
            title: 'Threat Distribution by Risk Level',
            xaxis: { title: 'Risk Level' },
            yaxis: { title: 'Number of Threats' }
        };
        
        Plotly.newPlot('threat-distribution-chart', threatData, layout);
    </script>
</body>
</html>
        """
        
        tech_template_path = self.template_dir / "technical_analysis.html"
        if not tech_template_path.exists():
            with open(tech_template_path, 'w', encoding='utf-8') as f:
                f.write(technical_template.strip())
    
    def _load_mitre_descriptions(self) -> Dict[str, str]:
        """Load MITRE ATT&CK technique descriptions"""
        return {
            'T1078': 'Valid Accounts - Adversaries may obtain and abuse credentials of existing accounts',
            'T1059.001': 'PowerShell - Adversaries may abuse PowerShell commands and scripts',
            'T1059.003': 'Windows Command Shell - Adversaries may abuse the Windows command shell',
            'T1021.002': 'SMB/Windows Admin Shares - Adversaries may use SMB to laterally move',
            'T1055': 'Process Injection - Adversaries may inject code into processes',
            'T1003': 'OS Credential Dumping - Adversaries may attempt to dump credentials',
            'T1053': 'Scheduled Task/Job - Adversaries may abuse task scheduling functionality',
            'T1112': 'Modify Registry - Adversaries may interact with the Windows Registry'
        }
    
    def load_inference_results(self, results_file: str) -> Dict[str, Any]:
        """Load inference results from JSON file"""
        logger.info(f"📄 Loading inference results from {results_file}")
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        required_fields = ['threat_predictions', 'risk_summary', 'timestamp']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        logger.info(f"✅ Loaded {len(data.get('threat_predictions', []))} threat predictions")
        return data
    
    def generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary report"""
        logger.info("📊 Generating executive summary report")
        
        # Extract key metrics
        risk_summary = data['risk_summary']
        threats = data['threat_predictions']
        
        # Generate risk description
        risk_descriptions = {
            'CRITICAL': 'Immediate action required. Critical threats detected that pose significant risk to organizational security.',
            'HIGH': 'High-priority threats detected. Recommend immediate investigation and response.',
            'MEDIUM': 'Moderate threats detected. Monitor closely and investigate as resources permit.',
            'LOW': 'Low-level threats detected. Continue monitoring with standard security procedures.'
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_summary, threats)
        
        # Prepare template data
        template_data = {
            'title': self.config.title,
            'organization': self.config.organization,
            'classification': self.config.classification,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_risk': risk_summary['overall_risk'],
            'risk_description': risk_descriptions.get(risk_summary['overall_risk'], 'Risk assessment completed.'),
            'total_threats': risk_summary['total_threats'],
            'critical_threats': risk_summary['critical_threats'],
            'high_threats': risk_summary['high_threats'],
            'entities_at_risk': risk_summary['entities_at_risk'],
            'top_threats': data.get('top_threats', []),
            'recommendations': recommendations,
            'processing_time': data.get('processing_time_seconds', 0)
        }
        
        # Render template
        template = self.jinja_env.get_template('executive_summary.html')
        html_content = template.render(**template_data)
        
        return html_content
    
    def generate_technical_report(self, data: Dict[str, Any]) -> str:
        """Generate technical analysis report"""
        logger.info("🔧 Generating technical analysis report")
        
        risk_summary = data['risk_summary']
        
        # Prepare template data
        template_data = {
            'title': self.config.title,
            'organization': self.config.organization,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_events': data.get('total_events', 0),
            'total_entities': data.get('total_entities', 0),
            'processing_time': data.get('processing_time_seconds', 0),
            'model_auc': data.get('model_info', {}).get('auc', 'N/A'),
            'critical_threats': risk_summary['critical_threats'],
            'high_threats': risk_summary['high_threats'],
            'medium_threats': risk_summary['medium_threats'],
            'low_threats': risk_summary['low_threats'],
            'mitre_techniques': data.get('mitre_techniques', []),
            'attack_chains': data.get('attack_chains', []),
            'detailed_threats': data.get('threat_predictions', [])
        }
        
        # Render template
        template = self.jinja_env.get_template('technical_analysis.html')
        html_content = template.render(**template_data)
        
        return html_content
    
    def _generate_recommendations(self, risk_summary: Dict[str, Any], 
                                 threats: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable security recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_summary['overall_risk'] == 'CRITICAL':
            recommendations.extend([
                "Immediately isolate affected systems and initiate incident response procedures",
                "Conduct emergency security assessment of all critical infrastructure",
                "Review and update access controls for high-privilege accounts"
            ])
        elif risk_summary['overall_risk'] == 'HIGH':
            recommendations.extend([
                "Prioritize investigation of high-risk threats within 24 hours",
                "Implement additional monitoring on affected systems",
                "Review security policies and access controls"
            ])
        
        # Threat-specific recommendations
        attack_types = set(t.get('edge_type', '') for t in threats)
        
        if 'executes' in attack_types:
            recommendations.append("Monitor process execution patterns and implement application whitelisting")
        
        if 'connects_to' in attack_types:
            recommendations.append("Review network segmentation and implement additional network monitoring")
        
        if 'logs_into' in attack_types:
            recommendations.append("Strengthen authentication mechanisms and implement multi-factor authentication")
        
        # General recommendations
        recommendations.extend([
            "Update threat intelligence feeds and security signatures",
            "Conduct security awareness training for affected user groups",
            "Review and test incident response procedures"
        ])
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def create_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create visualization charts"""
        logger.info("📈 Creating visualization charts")
        
        visualizations = {}
        
        # Risk distribution pie chart
        risk_summary = data['risk_summary']
        risk_data = [
            risk_summary['critical_threats'],
            risk_summary['high_threats'],
            risk_summary['medium_threats'],
            risk_summary['low_threats']
        ]
        
        if sum(risk_data) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['Critical', 'High', 'Medium', 'Low'],
                values=risk_data,
                marker_colors=['#d32f2f', '#f57c00', '#1976d2', '#388e3c']
            )])
            
            fig.update_layout(title="Threat Distribution by Risk Level")
            visualizations['risk_distribution'] = fig.to_html(include_plotlyjs='cdn')
        
        # Timeline of threats (if timestamp data available)
        threats = data.get('threat_predictions', [])
        if threats and any('timestamp' in t for t in threats):
            timestamps = [t.get('timestamp', '') for t in threats if 'timestamp' in t]
            risk_scores = [t.get('risk_score', 0) for t in threats if 'timestamp' in t]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=risk_scores,
                mode='markers+lines',
                name='Threat Risk Scores',
                marker=dict(size=8, color=risk_scores, colorscale='Reds')
            ))
            
            fig.update_layout(
                title="Threat Timeline",
                xaxis_title="Time",
                yaxis_title="Risk Score"
            )
            
            visualizations['threat_timeline'] = fig.to_html(include_plotlyjs='cdn')
        
        return visualizations
    
    def export_to_pdf(self, html_content: str, output_path: str) -> None:
        """Export HTML content to PDF"""
        try:
            # Use WeasyPrint for better PDF generation
            HTML(string=html_content).write_pdf(output_path)
            logger.info(f"📄 PDF report saved to: {output_path}")
        except Exception as e:
            logger.warning(f"⚠️  PDF generation failed: {e}")
            # Fallback: save as HTML
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"📄 HTML report saved to: {html_path}")
    
    def export_to_csv(self, data: Dict[str, Any], output_path: str) -> None:
        """Export threat data to CSV"""
        threats = data.get('threat_predictions', [])
        if not threats:
            logger.warning("⚠️  No threat data to export to CSV")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(threats)
        
        # Select key columns
        columns = ['prediction_id', 'source_entity', 'target_entity', 'edge_type', 
                  'risk_score', 'confidence', 'mitre_techniques', 'timestamp']
        
        df_export = df[[col for col in columns if col in df.columns]]
        
        # Export to CSV
        df_export.to_csv(output_path, index=False)
        logger.info(f"📊 CSV export saved to: {output_path}")
    
    def generate_report(self, results_file: str, output_prefix: Optional[str] = None) -> Dict[str, str]:
        """Generate complete report suite"""
        logger.info("🚀 Starting Ultimate THGNN Report Generation")
        start_time = time.time()
        
        try:
            # Load data
            data = self.load_inference_results(results_file)
            
            # Generate output prefix
            if output_prefix is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_prefix = f"thgnn_report_{timestamp}"
            
            generated_files = {}
            
            # Generate executive summary
            if self.config.include_executive_summary:
                exec_html = self.generate_executive_summary(data)
                
                if 'html' in self.config.output_formats:
                    exec_html_path = self.output_dir / f"{output_prefix}_executive.html"
                    with open(exec_html_path, 'w', encoding='utf-8') as f:
                        f.write(exec_html)
                    generated_files['executive_html'] = str(exec_html_path)
                
                if 'pdf' in self.config.output_formats:
                    exec_pdf_path = self.output_dir / f"{output_prefix}_executive.pdf"
                    self.export_to_pdf(exec_html, str(exec_pdf_path))
                    generated_files['executive_pdf'] = str(exec_pdf_path)
            
            # Generate technical report
            if self.config.include_technical_details:
                tech_html = self.generate_technical_report(data)
                
                if 'html' in self.config.output_formats:
                    tech_html_path = self.output_dir / f"{output_prefix}_technical.html"
                    with open(tech_html_path, 'w', encoding='utf-8') as f:
                        f.write(tech_html)
                    generated_files['technical_html'] = str(tech_html_path)
                
                if 'pdf' in self.config.output_formats:
                    tech_pdf_path = self.output_dir / f"{output_prefix}_technical.pdf"
                    self.export_to_pdf(tech_html, str(tech_pdf_path))
                    generated_files['technical_pdf'] = str(tech_pdf_path)
            
            # Export raw data
            if 'json' in self.config.output_formats:
                json_path = self.output_dir / f"{output_prefix}_data.json"
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                generated_files['json'] = str(json_path)
            
            if 'csv' in self.config.output_formats:
                csv_path = self.output_dir / f"{output_prefix}_threats.csv"
                self.export_to_csv(data, str(csv_path))
                generated_files['csv'] = str(csv_path)
            
            # Generate summary
            generation_time = time.time() - start_time
            
            logger.info("🎉 Report generation completed successfully!")
            logger.info(f"⏱️  Generation time: {generation_time:.2f} seconds")
            logger.info(f"📁 Output directory: {self.output_dir}")
            
            for report_type, file_path in generated_files.items():
                logger.info(f"📄 {report_type}: {file_path}")
            
            return generated_files
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            raise

def main():
    """Main report generation function"""
    parser = argparse.ArgumentParser(description='Ultimate THGNN Report Generator')
    parser.add_argument('--input', type=str, required=True, help='Path to inference results JSON file')
    parser.add_argument('--output', type=str, help='Output file prefix')
    parser.add_argument('--title', type=str, default='THGNN Threat Analysis Report', help='Report title')
    parser.add_argument('--organization', type=str, default='Security Operations Center', help='Organization name')
    parser.add_argument('--classification', type=str, default='CONFIDENTIAL', help='Security classification')
    parser.add_argument('--formats', nargs='+', default=['html', 'pdf', 'json'], 
                       choices=['html', 'pdf', 'json', 'csv'], help='Output formats')
    parser.add_argument('--no-executive', action='store_true', help='Skip executive summary')
    parser.add_argument('--no-technical', action='store_true', help='Skip technical report')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = ReportConfig(
            title=args.title,
            organization=args.organization,
            classification=args.classification,
            include_executive_summary=not args.no_executive,
            include_technical_details=not args.no_technical,
            output_formats=args.formats
        )
        
        # Create report generator
        generator = UltimateTHGNNReportGenerator(config)
        
        # Generate reports
        generated_files = generator.generate_report(args.input, args.output)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 ULTIMATE THGNN REPORT GENERATION COMPLETED!")
        print("="*80)
        print(f"📄 Input file: {args.input}")
        print(f"📁 Output directory: {generator.output_dir}")
        print(f"📊 Generated {len(generated_files)} report files:")
        
        for report_type, file_path in generated_files.items():
            print(f"   • {report_type.upper()}: {Path(file_path).name}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()