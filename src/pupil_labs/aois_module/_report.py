import io
from functools import partial
from pathlib import Path

from reportlab.lib import pagesizes
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Table,
)


def generate_report(self):
    """
    Generate a PDF report with the results of the AOI analysis.
    """
    filename = self.output_path + "/report_AOIs.pdf"
    _, page_width = pagesizes.A4

    def header(canvas, doc, content):
        canvas.saveState()
        w, h = content.wrap(doc.width, doc.topMargin)
        content.drawOn(
            canvas, doc.leftMargin, doc.height + doc.bottomMargin + doc.topMargin - h
        )
        canvas.restoreState()

    def footer(canvas, doc, content):
        canvas.saveState()
        w, h = content.wrap(doc.width, doc.bottomMargin)
        content.drawOn(canvas, doc.leftMargin, h)
        canvas.restoreState()

    def header_and_footer(canvas, doc, header_content, footer_content):
        header(canvas, doc, header_content)
        footer(canvas, doc, footer_content)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        filename,
        pagesize=pagesizes.portrait(pagesizes.A4),
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2.5 * cm,
    )

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")

    title_text = "<b>Areas of Interest Report</b>"
    title = Paragraph(title_text, styles["h2"])

    logo_path = Path(__file__).parent / "assets" / "primary.png"
    logo = Image(logo_path, width=100, height=100)

    header_content = Table(
        [[logo, title]], colWidths=[2 * cm, 14 * cm], rowHeights=[2 * cm]
    )

    footer_content = Paragraph(
        "This is a footer. It goes on every page.  ", styles["Normal"]
    )

    template = PageTemplate(
        id="test",
        frames=frame,
        onPage=partial(
            header_and_footer,
            header_content=header_content,
            footer_content=footer_content,
        ),
    )

    doc.addPageTemplates([template])

    # Build the story
    story = []

    # Add images to the report
    for key, value in self.figure.items():
        img = Image(io.BytesIO(value))
        if key == "AOIs":
            img.drawWidth = page_width - (400)
            img.drawHeight = img.drawWidth * img.imageHeight / img.imageWidth
            story.append(img)
            story.append(PageBreak())
        else:
            img.drawWidth = page_width - (300)
            img.drawHeight = img.drawWidth * img.imageHeight / img.imageWidth
            story.append(img)
    story.append(PageBreak())
    # Add tables to the report

    # Build the PDF report
    doc.build(story)


if __name__ == "__main__":
    from pupil_labs.aois_module._defineAOIs import defineAOIs

    self = defineAOIs()
    self.output_path = "/Users/mgg/Desktop"
    self.figure = {}
    generate_report(self)
