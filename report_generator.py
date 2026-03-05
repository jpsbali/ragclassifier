import io
from datetime import datetime
from typing import List, Optional, Any

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from risk_evaluator import RiskEvaluator, TCHClassification
from src.models import SupervisorDecision


def generate_pdf_report(
    run_id: str, 
    results: List[SupervisorDecision], 
    risk_evaluator: Optional[RiskEvaluator]
) -> bytes:
    """
    Generates a PDF summary report for the classification run.
    Returns bytes of the PDF file.
    """
    if not HAS_REPORTLAB:
        return b""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # --- Title Section ---
    title_style = styles['Title']
    elements.append(Paragraph("Classification Run Report", title_style))
    
    normal_style = styles['Normal']
    elements.append(Paragraph(f"Run ID: {run_id}", normal_style))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 12))

    # --- Executive Summary ---
    h2_style = styles['Heading2']
    elements.append(Paragraph("Executive Summary", h2_style))
    
    total_docs = len(results)
    high_risk_count = 0
    high_priority_count = 0
    class_counts = {}
    total_cost = 0.0
    
    # Prepare data for the detailed table while calculating stats
    table_data = [["Document Name", "Class", "Conf", "Risk", "Cost"]]
    
    for d in results:
        is_hr_by_classifier = d.classification.value == TCHClassification.HUMAN_REVIEW.value
        is_high_priority = d.review_priority == "HIGH"

        final_class = d.classification.value
        risk_flag = "Low"
        
        if risk_evaluator:
            risk_eval = risk_evaluator.calculate_risk(
                TCHClassification(d.classification.value), d.confidence
            )
            if is_hr_by_classifier or risk_eval.is_high_risk:
                final_class = TCHClassification.HUMAN_REVIEW.value
                risk_flag = "HIGH"
                high_risk_count += 1
                if is_high_priority:
                    high_priority_count += 1
        
        class_counts[final_class] = class_counts.get(final_class, 0) + 1
        total_cost += d.estimated_cost
        
        # Truncate long document names for the PDF table
        doc_name = (d.document_name[:35] + '..') if len(d.document_name) > 35 else d.document_name
        
        table_data.append([
            doc_name,
            final_class,
            f"{d.confidence:.2f}",
            risk_flag,
            f"${d.estimated_cost:.4f}"
        ])

    summary_text = f"<b>Total Documents Processed:</b> {total_docs}<br/>"
    summary_text += f"<b>Total Estimated Cost:</b> ${total_cost:.4f}<br/>"
    if risk_evaluator:
        summary_text += f"<b>Human Review Required:</b> {high_risk_count}<br/>"
        summary_text += f"<b>High-Priority Review:</b> {high_priority_count}<br/>"
    
    summary_text += "<br/><b>Classification Distribution:</b><br/>"
    for cls, count in class_counts.items():
        summary_text += f"- {cls}: {count}<br/>"

    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 12))

    # --- Detailed Results Table ---
    elements.append(Paragraph("Detailed Results", h2_style))
    
    # Table Styling
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#262730')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ])
    
    # Highlight High Risk rows in Red
    for i, row in enumerate(table_data):
        if i > 0 and row[3] == "HIGH":  # Skip header
            t_style.add('TEXTCOLOR', (0, i), (-1, i), colors.red)
            t_style.add('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold')

    t = Table(table_data, colWidths=[220, 100, 50, 50, 60])
    t.setStyle(t_style)
    elements.append(t)

    doc.build(elements)
    return buffer.getvalue()