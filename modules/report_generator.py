"""
PDF Report Generator Module
Creates comprehensive PDF reports with executive summary, maps, and recommendations
"""

import os
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import pandas as pd


class ReportGenerator:
    """
    Generates comprehensive PDF reports for business location analysis.
    Includes executive summary, maps, scoring details, and recommendations.
    """
    
    def __init__(self, output_path: str = 'data/analysis_report.pdf'):
        """
        Initialize the ReportGenerator.
        
        Args:
            output_path: Path for the output PDF file
        """
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f77b4')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=13,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e')
        ))
        # Safely configure BodyText without redefining existing style
        if 'BodyText' in getattr(self.styles, 'byName', {}):
            body = self.styles['BodyText']
            body.fontSize = 11
            body.spaceAfter = 12
            body.alignment = TA_JUSTIFY
        else:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_JUSTIFY
            ))
        
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor('#27ae60'),
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(self, 
                       city: str,
                       business_type: str,
                       radius_km: float,
                       results: dict,
                       weights: dict,
                       top_locations: pd.DataFrame) -> str:
        """
        Generate a comprehensive PDF report.
        
        Args:
            city: City name analyzed
            business_type: Type of business
            radius_km: Search radius used
            results: Analysis results dictionary
            weights: Scoring weights used
            top_locations: DataFrame of top recommended locations
            
        Returns:
            Path to the generated PDF file
        """
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        story.extend(self._create_title_page(city, business_type, radius_km))
        
        story.extend(self._create_executive_summary(
            city, business_type, radius_km, results, weights
        ))
        
        story.extend(self._create_methodology_section(weights))
        
        story.extend(self._create_analysis_results(results))
        
        story.extend(self._create_recommendations_section(top_locations, business_type))
        
        story.extend(self._create_conclusion(city, business_type, top_locations))
        
        doc.build(story)
        
        return self.output_path
    
    def _create_title_page(self, city: str, business_type: str, 
                           radius_km: float) -> list:
        """Create the title page elements."""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        
        elements.append(Paragraph(
            "Business Location Analysis Report",
            self.styles['CustomTitle']
        ))
        
        elements.append(Spacer(1, 0.3*inch))
        
        subtitle = f"Geospatial Clustering & Location Recommendation<br/>" \
                   f"for <b>{business_type.title()}</b> in <b>{city}</b>"
        elements.append(Paragraph(subtitle, self.styles['Heading2']))
        
        elements.append(Spacer(1, 1*inch))
        
        report_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}<br/>
        <b>Analysis Area:</b> {radius_km} km radius from city center<br/>
        <b>Business Type:</b> {business_type.title()}<br/>
        <b>City:</b> {city}
        """
        elements.append(Paragraph(report_info, self.styles['BodyText']))
        
        elements.append(PageBreak())
        
        return elements
    
    def _create_executive_summary(self, city: str, business_type: str,
                                  radius_km: float, results: dict,
                                  weights: dict) -> list:
        """Create the executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        summary_text = f"""
        This report presents a comprehensive geospatial analysis for identifying 
        optimal locations to establish a new <b>{business_type}</b> business in 
        <b>{city}</b>. The analysis covers a {radius_km} km radius from the city 
        center and utilizes data from OpenStreetMap to evaluate potential sites 
        based on multiple factors including demand, competition, accessibility, 
        and infrastructure.
        """
        elements.append(Paragraph(summary_text, self.styles['BodyText']))
        
        elements.append(Paragraph("Key Findings", self.styles['SubSection']))
        
        cluster_stats = results.get('cluster_stats', {})
        scoring_report = results.get('scoring_report', {})
        
        findings = [
            f"<b>{results.get('cleaned_poi_count', 0)}</b> Points of Interest analyzed",
            f"<b>{cluster_stats.get('n_clusters', 0)}</b> distinct business clusters identified",
            f"<b>{len(results.get('hotspots', []))}</b> high-density hotspot areas detected",
            f"<b>{scoring_report.get('total_competitors', 0)}</b> existing competitors found",
            f"<b>{scoring_report.get('total_candidates', 0)}</b> candidate locations evaluated",
            f"Top location score: <b>{scoring_report.get('top_score', 0):.3f}</b>"
        ]
        
        for finding in findings:
            elements.append(Paragraph(f"• {finding}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_methodology_section(self, weights: dict) -> list:
        """Create the methodology section."""
        elements = []
        
        elements.append(Paragraph("Methodology", self.styles['SectionHeader']))
        
        method_text = """
        The analysis employs a multi-factor scoring model that evaluates each 
        candidate location based on four key dimensions:
        """
        elements.append(Paragraph(method_text, self.styles['BodyText']))
        
        elements.append(Paragraph("Scoring Weights", self.styles['SubSection']))
        
        weight_data = [
            ['Factor', 'Weight', 'Description'],
            ['Demand', f"{weights.get('demand', 0.4)*100:.0f}%", 
             'Based on supporting POIs and foot traffic indicators'],
            ['Competition', f"{weights.get('competition', 0.3)*100:.0f}%", 
             'Inverse of competitor density (lower = better)'],
            ['Accessibility', f"{weights.get('accessibility', 0.2)*100:.0f}%", 
             'Transport links and road connectivity'],
            ['Infrastructure', f"{weights.get('infrastructure', 0.1)*100:.0f}%", 
             'Nearby essential services (banks, hospitals, schools)']
        ]
        
        table = Table(weight_data, colWidths=[1.5*inch, 1*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph("Algorithms Used", self.styles['SubSection']))
        
        algo_text = """
        <b>DBSCAN Clustering:</b> Density-Based Spatial Clustering with haversine 
        distance metric for identifying natural groupings of POIs.<br/><br/>
        <b>Kernel Density Estimation:</b> For creating smooth heatmaps of POI 
        distribution and competition density.<br/><br/>
        <b>K-D Tree Spatial Indexing:</b> For efficient nearest-neighbor queries 
        when computing proximity scores.
        """
        elements.append(Paragraph(algo_text, self.styles['BodyText']))
        
        return elements
    
    def _create_analysis_results(self, results: dict) -> list:
        """Create the analysis results section."""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("Analysis Results", self.styles['SectionHeader']))
        
        elements.append(Paragraph("Cluster Analysis", self.styles['SubSection']))
        
        cluster_stats = results.get('cluster_stats', {})
        cluster_text = f"""
        The spatial clustering analysis identified <b>{cluster_stats.get('n_clusters', 0)}</b> 
        distinct clusters of Points of Interest within the search area. 
        <b>{cluster_stats.get('n_noise', 0)}</b> POIs were classified as noise 
        (isolated points not belonging to any cluster).
        """
        elements.append(Paragraph(cluster_text, self.styles['BodyText']))
        
        if 'clusters' in cluster_stats and cluster_stats['clusters']:
            elements.append(Paragraph("Top Clusters by Size", self.styles['SubSection']))
            
            cluster_data = [['Cluster ID', 'POI Count', 'Avg Radius (km)', 'Density']]
            
            sorted_clusters = sorted(
                cluster_stats['clusters'].items(),
                key=lambda x: x[1]['size'],
                reverse=True
            )[:5]
            
            for label, data in sorted_clusters:
                cluster_data.append([
                    str(label),
                    str(data['size']),
                    f"{data['avg_radius_km']:.2f}",
                    f"{data['density']:.1f}"
                ])
            
            table = Table(cluster_data, colWidths=[1.2*inch, 1.2*inch, 1.5*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.2*inch))
        
        hotspots = results.get('hotspots', [])
        if hotspots:
            elements.append(Paragraph("Hotspot Areas", self.styles['SubSection']))
            
            hotspot_text = f"""
            <b>{len(hotspots)}</b> high-density hotspot areas were identified. 
            These areas show significant concentration of business activity and 
            may represent both opportunities (high foot traffic) and challenges 
            (intense competition).
            """
            elements.append(Paragraph(hotspot_text, self.styles['BodyText']))
        
        return elements
    
    def _create_recommendations_section(self, top_locations: pd.DataFrame,
                                        business_type: str) -> list:
        """Create the recommendations section."""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("Recommended Locations", self.styles['SectionHeader']))
        
        intro_text = f"""
        Based on the multi-factor analysis, the following locations are recommended 
        for establishing a new <b>{business_type}</b> business. Locations are ranked 
        by their composite score, with higher scores indicating more favorable conditions.
        """
        elements.append(Paragraph(intro_text, self.styles['BodyText']))
        
        if top_locations is not None and len(top_locations) > 0:
            elements.append(Paragraph("Top 10 Recommended Locations", self.styles['SubSection']))
            
            loc_data = [['Rank', 'Latitude', 'Longitude', 'Final Score', 
                        'Demand', 'Competition', 'Access.', 'Infra.']]
            
            for idx, row in top_locations.head(10).iterrows():
                loc_data.append([
                    str(int(row['rank'])),
                    f"{row['latitude']:.4f}",
                    f"{row['longitude']:.4f}",
                    f"{row['final_score']:.3f}",
                    f"{row['demand_score']:.2f}",
                    f"{row['competition_score']:.2f}",
                    f"{row['accessibility_score']:.2f}",
                    f"{row['infrastructure_score']:.2f}"
                ])
            
            table = Table(loc_data, colWidths=[0.5*inch, 0.9*inch, 0.9*inch, 0.8*inch,
                                               0.7*inch, 0.9*inch, 0.7*inch, 0.6*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (0, 3), colors.HexColor('#d5f5e3')),
                ('BACKGROUND', (0, 4), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3*inch))
            
            elements.append(Paragraph("Top 3 Location Details", self.styles['SubSection']))
            
            for idx, row in top_locations.head(3).iterrows():
                rank = int(row['rank'])
                medal = "Gold" if rank == 1 else "Silver" if rank == 2 else "Bronze"
                
                detail_text = f"""
                <b>#{rank} - {medal} Location</b><br/>
                Coordinates: ({row['latitude']:.5f}, {row['longitude']:.5f})<br/>
                This location scores highest in 
                {'demand' if row['demand_score'] == max(row['demand_score'], row['competition_score'], row['accessibility_score'], row['infrastructure_score']) else 'multiple factors'}, 
                with a final composite score of <b>{row['final_score']:.3f}</b>.
                """
                elements.append(Paragraph(detail_text, self.styles['BodyText']))
        
        return elements
    
    def _create_conclusion(self, city: str, business_type: str,
                          top_locations: pd.DataFrame) -> list:
        """Create the conclusion section."""
        elements = []
        
        elements.append(Paragraph("Conclusion", self.styles['SectionHeader']))
        
        if top_locations is not None and len(top_locations) > 0:
            top_score = top_locations['final_score'].max()
            avg_score = top_locations.head(10)['final_score'].mean()
            
            conclusion_text = f"""
            This geospatial analysis has identified multiple viable locations for 
            establishing a <b>{business_type}</b> business in <b>{city}</b>. 
            The top recommended location achieves a score of <b>{top_score:.3f}</b>, 
            with the top 10 locations averaging <b>{avg_score:.3f}</b>.
            <br/><br/>
            Key recommendations:
            """
            elements.append(Paragraph(conclusion_text, self.styles['BodyText']))
            
            recommendations = [
                "Consider the top 3 locations for detailed site visits",
                "Evaluate local real estate availability and costs at recommended coordinates",
                "Conduct on-ground competitor analysis to validate the competition scores",
                "Assess foot traffic patterns at different times of day",
                "Consider seasonal variations in business activity"
            ]
            
            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        disclaimer = """
        <i>Note: This analysis is based on OpenStreetMap data and algorithmic scoring. 
        Actual business viability depends on many factors not captured in this model, 
        including local regulations, real estate costs, and market conditions. 
        Professional consultation is recommended before making investment decisions.</i>
        """
        elements.append(Paragraph(disclaimer, self.styles['BodyText']))
        
        return elements
    
    def generate_quick_summary(self, results: dict, 
                               top_locations: pd.DataFrame) -> str:
        """
        Generate a quick text summary of results.
        
        Args:
            results: Analysis results dictionary
            top_locations: DataFrame of top locations
            
        Returns:
            Summary string
        """
        cluster_stats = results.get('cluster_stats', {})
        scoring_report = results.get('scoring_report', {})
        
        summary = []
        summary.append("=" * 50)
        summary.append("QUICK ANALYSIS SUMMARY")
        summary.append("=" * 50)
        summary.append(f"POIs Analyzed: {results.get('cleaned_poi_count', 0)}")
        summary.append(f"Clusters Found: {cluster_stats.get('n_clusters', 0)}")
        summary.append(f"Hotspots: {len(results.get('hotspots', []))}")
        summary.append(f"Competitors: {scoring_report.get('total_competitors', 0)}")
        summary.append(f"Candidates Evaluated: {scoring_report.get('total_candidates', 0)}")
        summary.append("")
        summary.append("TOP 5 LOCATIONS:")
        
        if top_locations is not None:
            for idx, row in top_locations.head(5).iterrows():
                summary.append(
                    f"  #{int(row['rank'])}. ({row['latitude']:.4f}, {row['longitude']:.4f}) "
                    f"Score: {row['final_score']:.3f}"
                )
        
        summary.append("=" * 50)
        
        return "\n".join(summary)
