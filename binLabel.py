import streamlit as st
import pandas as pd
import os
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader
from io import BytesIO
import subprocess
import sys
import tempfile
import base64
import re

# Define sticker dimensions
STICKER_WIDTH = 10 * cm
STICKER_HEIGHT = 15 * cm
STICKER_PAGESIZE = (STICKER_WIDTH, STICKER_HEIGHT)

# Define content box dimensions
CONTENT_BOX_WIDTH = 10 * cm  # Same width as page
CONTENT_BOX_HEIGHT = 7.2 * cm  # Half the page height

# Check for PIL and install if needed
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.error("PIL not available. Please install: pip install pillow")
    st.stop()

# Check for QR code library and install if needed
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    st.error("qrcode not available. Please install: pip install qrcode")
    st.stop()

# Define paragraph styles
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

def get_dynamic_description_style(description_text):
    """
    Create a dynamic description style based on text length
    Adjusts font size and leading for longer descriptions to fit within the box
    """
    text_length = len(description_text)
    
    if text_length <= 50:
        # Short description - use standard size
        return ParagraphStyle(
            name='Description', 
            fontName='Helvetica', 
            fontSize=11, 
            alignment=TA_CENTER, 
            leading=12
        )
    elif text_length <= 80:
        # Medium description - slightly smaller
        return ParagraphStyle(
            name='DescriptionMedium', 
            fontName='Helvetica', 
            fontSize=10, 
            alignment=TA_CENTER, 
            leading=10
        )
    elif text_length <= 120:
        # Long description - smaller font
        return ParagraphStyle(
            name='DescriptionLong', 
            fontName='Helvetica', 
            fontSize=9, 
            alignment=TA_CENTER, 
            leading=9
        )
    else:
        # Very long description - smallest font
        return ParagraphStyle(
            name='DescriptionVeryLong', 
            fontName='Helvetica', 
            fontSize=8, 
            alignment=TA_CENTER, 
            leading=8
        )

def truncate_description_smartly(description, max_length=150):
    """
    Smart truncation that tries to break at word boundaries
    """
    if len(description) <= max_length:
        return description
    
    # Try to truncate at last complete word within limit
    truncated = description[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If last space is reasonably close to limit
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."

def generate_qr_code(data_string):
    """
    Generate a QR code from the given data string
    """
    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        
        # Add data
        qr.add_data(data_string)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to bytes that reportlab can use
        img_buffer = BytesIO()
        qr_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create a QR code image with specified size
        return Image(img_buffer, width=2.5*cm, height=2.5*cm)
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        return None

def parse_location_string(location_str):
    """Parse a location string into components for table display"""
    # Initialize with empty values
    location_parts = [''] * 7

    if not location_str or not isinstance(location_str, str):
        return location_parts

    # Remove any extra spaces
    location_str = location_str.strip()

    # Try to parse location components
    pattern = r'([^_\s]+)'
    matches = re.findall(pattern, location_str)

    # Fill the available parts
    for i, match in enumerate(matches[:7]):
        location_parts[i] = match

    return location_parts

def detect_bus_model_columns(df_columns):
    """
    Improved bus model column detection with better pattern matching
    Returns a dictionary mapping column names to bus models
    """
    bus_model_columns = {}
    
    for col in df_columns:
        col_str = str(col).strip()
        col_upper = col_str.upper()
        
        # More precise pattern matching to avoid false positives
        # Check for 7M patterns
        if re.search(r'\b7\s*M\b', col_upper) or re.search(r'\bSEVEN\s*M\b', col_upper):
            bus_model_columns[col] = '7M'
        # Check for 9M patterns  
        elif re.search(r'\b9\s*M\b', col_upper) or re.search(r'\bNINE\s*M\b', col_upper):
            bus_model_columns[col] = '9M'
        # Check for 12M patterns
        elif re.search(r'\b12\s*M\b', col_upper) or re.search(r'\bTWELVE\s*M\b', col_upper):
            bus_model_columns[col] = '12M'
        # Handle QTY/VEH columns with specific bus model indicators
        elif 'QTY' in col_upper and 'VEH' in col_upper:
            # Check which bus model this QTY/VEH column refers to
            if re.search(r'\b7\b', col_upper) and not re.search(r'\b12\b', col_upper):
                bus_model_columns[col] = '7M'
            elif re.search(r'\b9\b', col_upper):
                bus_model_columns[col] = '9M'
            elif re.search(r'\b12\b', col_upper):
                bus_model_columns[col] = '12M'
    
    return bus_model_columns

def extract_bus_model_quantities(row, bus_model_columns):
    """
    Extract quantities for 7M, 9M, and 12M based on bus model columns
    Returns a dictionary with keys '7M', '9M', '12M' and their respective quantities
    """
    quantities = {'7M': '', '9M': '', '12M': ''}
    
    # Debug: Print column mapping for troubleshooting
    if hasattr(st, 'session_state') and 'debug_mode' in st.session_state and st.session_state.debug_mode:
        st.write(f"Bus model columns detected: {bus_model_columns}")
    
    for col_name, bus_model in bus_model_columns.items():
        if col_name in row and pd.notna(row[col_name]):
            qty_value = str(row[col_name]).strip()
            # Only add non-zero, non-empty values
            if qty_value and qty_value not in ['0', '0.0', 'nan', 'NaN', '']:
                quantities[bus_model] = qty_value
    
    return quantities

def generate_sticker_labels(df, progress_bar=None, status_container=None):
    """Generate sticker labels with QR code from DataFrame"""
    
    # Create temporary file for PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        output_pdf_path = tmp_file.name

    # Create a function to draw the border box around content
    def draw_border(canvas, doc):
        canvas.saveState()
        # Draw border box around the content area (10cm x 7.5cm)
        # Position it at the top of the page with minimal margin
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm  # Position at top with minimal margin
        canvas.setStrokeColor(colors.Color(0, 0, 0, alpha=0.95))  # Slightly darker black (95% opacity)
        canvas.setLineWidth(1.8)  # Slightly thicker border
        canvas.rect(
            x_offset + doc.leftMargin,
            y_offset,
            CONTENT_BOX_WIDTH - 0.2*cm,  # Account for margins
            CONTENT_BOX_HEIGHT
        )
        canvas.restoreState()

    # Identify columns (case-insensitive)
    original_columns = df.columns.tolist()
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    # Find relevant columns
    part_no_col = next((col for col in cols if 'PART' in str(col) and ('NO' in str(col) or 'NUM' in str(col) or '#' in str(col))),
                   next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))

    desc_col = next((col for col in cols if 'DESC' in str(col)),
                   next((col for col in cols if 'NAME' in str(col)), cols[1] if len(cols) > 1 else part_no_col))

    # Look specifically for "QTY/BIN" column first, then fall back to general QTY column
    qty_bin_col = next((col for col in cols if 'QTY/BIN' in str(col) or 'QTY_BIN' in str(col) or 'QTYBIN' in str(col)), 
                  next((col for col in cols if 'QTY' in str(col) and 'BIN' in str(col)), None))
    
    # If no specific QTY/BIN column is found, fall back to general QTY column
    if not qty_bin_col:
        qty_bin_col = next((col for col in cols if 'QTY' in str(col) and 'VEH' not in str(col)),
                      next((col for col in cols if 'QUANTITY' in str(col)), None))
  
    loc_col = next((col for col in cols if 'LOC' in str(col) or 'POS' in str(col) or 'LOCATION' in str(col)),
                   cols[2] if len(cols) > 2 else desc_col)

    # Detect bus model columns using original column names to preserve case sensitivity
    bus_model_columns = detect_bus_model_columns(original_columns)
    
    # Convert bus model column keys to uppercase for consistency
    bus_model_columns_upper = {}
    for orig_col, bus_model in bus_model_columns.items():
        upper_col = orig_col.upper() if isinstance(orig_col, str) else orig_col
        bus_model_columns_upper[upper_col] = bus_model

    # Look for store location column
    store_loc_col = next((col for col in cols if 'STORE' in str(col) and 'LOC' in str(col)),
                      next((col for col in cols if 'STORELOCATION' in str(col)), None))

    if status_container:
        status_container.write(f"**Column Mapping:**")
        status_container.write(f"- Part No: {part_no_col}")
        status_container.write(f"- Description: {desc_col}")
        status_container.write(f"- Location: {loc_col}")
        status_container.write(f"- Qty/Bin: {qty_bin_col}")
        if bus_model_columns_upper:
            status_container.write(f"- Bus Model Columns: {bus_model_columns_upper}")
        if store_loc_col:
            status_container.write(f"- Store Location: {store_loc_col}")

    # Create document with minimal margins
    doc = SimpleDocTemplate(output_pdf_path, pagesize=STICKER_PAGESIZE,
                          topMargin=0.2*cm,  # Minimal top margin
                          bottomMargin=(STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm),  # Adjust bottom margin accordingly
                          leftMargin=0.1*cm, rightMargin=0.1*cm)

    content_width = CONTENT_BOX_WIDTH - 0.2*cm
    all_elements = []

    # Process each row as a single sticker
    total_rows = len(df)
    for index, row in df.iterrows():
        # Update progress
        if progress_bar:
            progress_bar.progress((index + 1) / total_rows)
        
        elements = []

        # Extract data
        part_no = str(row[part_no_col])
        desc = str(row[desc_col])
        
        # Smart truncation and dynamic styling for description
        processed_desc = truncate_description_smartly(desc, 150)
        dynamic_desc_style = get_dynamic_description_style(processed_desc)
        
        # Extract QTY/BIN properly
        qty_bin = ""
        if qty_bin_col and qty_bin_col in row and pd.notna(row[qty_bin_col]):
            qty_bin = str(row[qty_bin_col])
            
        # Extract bus model quantities
        bus_quantities = extract_bus_model_quantities(row, bus_model_columns_upper)
        
        location_str = str(row[loc_col]) if loc_col and loc_col in row else ""
        store_location = str(row[store_loc_col]) if store_loc_col and store_loc_col in row else ""
        location_parts = parse_location_string(location_str)

        # Generate QR code with part information
        qr_data = f"Part No: {part_no}\nDescription: {desc}\nLocation: {location_str}\n"
        qr_data += f"Store Location: {store_location}\n"
        qr_data += f"7M Qty: {bus_quantities['7M']}\n9M Qty: {bus_quantities['9M']}\n12M Qty: {bus_quantities['12M']}\n"
        qr_data += f"QTY/BIN: {qty_bin}"
        
        qr_image = generate_qr_code(qr_data)
        
        # Define row heights
        header_row_height = 0.9*cm
        desc_row_height = 1.0*cm
        qty_row_height = 0.5*cm
        location_row_height = 0.5*cm

        # Main table data with dynamic description styling
        main_table_data = [
            ["Part No", Paragraph(f"{part_no}", bold_style)],
            ["Description", Paragraph(processed_desc, dynamic_desc_style)],
            ["Qty/Bin", Paragraph(str(qty_bin), qty_style)]
        ]

        # Create main table
        main_table = Table(main_table_data,
                         colWidths=[content_width/3, content_width*2/3],
                         rowHeights=[header_row_height, desc_row_height, qty_row_height])

        main_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(main_table)

        # Store Location section
        store_loc_label = Paragraph("Store Location", ParagraphStyle(
            name='StoreLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))

        # Total width for the 7 inner columns (2/3 of full content width)
        inner_table_width = content_width * 2 / 3
        
        # Define proportional widths - same as Line Location for consistency
        col_proportions = [1.5, 2, 0.9, 0.9, 1, 1, 0.6]
        total_proportion = sum(col_proportions)
        
        # Calculate column widths based on proportions 
        inner_col_widths = [w * inner_table_width / total_proportion for w in col_proportions]

        # Use store_location if available, otherwise use empty values
        store_loc_values = parse_location_string(store_location) if store_location else ["", "", "", "", "", "", ""]

        store_loc_inner_table = Table(
            [store_loc_values],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )

        store_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),  # Make store location values bold
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        store_loc_table = Table(
            [[store_loc_label, store_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        store_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(store_loc_table)

        # Line Location section
        line_loc_label = Paragraph("Line Location", ParagraphStyle(
            name='LineLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))
        
        # Create the inner table
        line_loc_inner_table = Table(
            [location_parts],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )
        
        line_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),  # Make line location values bold
            ('FONTSIZE', (0, 0), (-1, -1), 9)
        ]))
        
        # Wrap the label and the inner table in a containing table
        line_loc_table = Table(
            [[line_loc_label, line_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        line_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(line_loc_table)

        # Add smaller spacer between line location and bottom section
        elements.append(Spacer(1, 0.3*cm))

        # Bottom section - Bus model quantities with appropriate placement
        mtm_box_width = 1.2*cm
        mtm_row_height = 1.5*cm

        # Create position matrix with bus model quantities in appropriate boxes
        position_matrix_data = [
            ["7M", "9M", "12M"],
            [
                Paragraph(f"<b>{bus_quantities['7M']}</b>", ParagraphStyle(
                    name='Bold7M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if bus_quantities['7M'] else "",
                Paragraph(f"<b>{bus_quantities['9M']}</b>", ParagraphStyle(
                    name='Bold9M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if bus_quantities['9M'] else "",
                Paragraph(f"<b>{bus_quantities['12M']}</b>", ParagraphStyle(
                    name='Bold12M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if bus_quantities['12M'] else ""
            ]
        ]

        mtm_table = Table(
            position_matrix_data,
            colWidths=[mtm_box_width, mtm_box_width, mtm_box_width],
            rowHeights=[mtm_row_height/2, mtm_row_height/2]
        )

        mtm_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        # QR code with preserved size
        qr_width = 2.5*cm
        qr_height = 2.5*cm

        if qr_image:
            qr_table = Table(
                [[qr_image]],
                colWidths=[qr_width],
                rowHeights=[qr_height]
            )
        else:
            qr_table = Table(
                [[Paragraph("QR", ParagraphStyle(
                    name='QRPlaceholder', fontName='Helvetica-Bold', fontSize=12, alignment=TA_CENTER
                ))]],
                colWidths=[qr_width],
                rowHeights=[qr_height]
            )

        qr_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Adjust spacing for better layout
        left_spacer_width = 0.8*cm
        right_spacer_width = content_width - 3*mtm_box_width - qr_width - left_spacer_width

        # Combine MTM boxes and QR code in one row with better spacing
        bottom_row = Table(
            [[mtm_table, "", qr_table, ""]],
            colWidths=[3*mtm_box_width, left_spacer_width, qr_width, right_spacer_width],
            rowHeights=[qr_height]
        )

        bottom_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(bottom_row)

        # Add all elements for this sticker to the document
        all_elements.extend(elements)

        # Add page break after each sticker (except the last one)
        if index < len(df) - 1:
            all_elements.append(PageBreak())

    # Build the document
    try:
        # Pass the draw_border function to build to add border box
        doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
        return output_pdf_path
    except Exception as e:
        st.error(f"Error building PDF: {e}")
        return None

def get_download_link(file_path, filename):
    """Generate a download link for the PDF file"""
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF</a>'
    return href

def main():
    st.set_page_config(
        page_title="Bin Label Generator", 
        page_icon="🏷️", 
        layout="wide"
    )
    
    st.title("🏷️ Bin Label Generator")
    st.markdown("Generate professional bin labels with QR codes from your Excel/CSV data")
    
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", help="Enable to see bus model column detection details")
    if debug_mode:
        st.session_state.debug_mode = True
    
    # Sidebar for information
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload your Excel or CSV file
        2. Preview the data to ensure correct formatting
        3. Click 'Generate Labels' to create PDF
        4. Download the generated PDF file
        
        **Required Columns:**
        - Part Number (PART NO, PARTNO, etc.)
        - Description (DESC, NAME, etc.)
        - Location (LOC, LOCATION, etc.)
        - Quantity/Bin (QTY/BIN, QTY, etc.)
        
        **Bus Model Columns:**
        - Columns containing "7M", "9M", or "12M"
        - QTY/VEH columns with bus model indicators
        - Will be automatically detected and mapped
        """)
        
        st.header("✨ Features")
        st.markdown("""
        - **Improved Bus Model Detection**
        - **Automatic QR Code Generation**
        - **Smart Description Sizing**
        - **Professional Layout**
        - **Border Box Design**
        - **Multiple File Formats**
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose your Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload the file containing your bin label data"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
                
                # Display file information
                st.subheader("📊 Data Preview")
                st.write(f"**Columns found:** {', '.join(df.columns.tolist())}")
                
                # Show bus model column detection if debug mode is on
                if debug_mode:
                    bus_model_cols = detect_bus_model_columns(df.columns.tolist())
                    if bus_model_cols:
                        st.write("**🚌 Bus Model Columns Detected:**")
                        for col, model in bus_model_cols.items():
                            st.write(f"- {col} → {model}")
                    else:
                        st.write("**🚌 No Bus Model Columns Detected**")
                
                # Show first few rows
                st.dataframe(df.head(10), use_container_width=True)
                
                if len(df) > 10:
                    st.info(f"Showing first 10 rows. Total rows: {len(df)}")
                
                # Generate labels button
                st.subheader("🚀 Generate Labels")
                
                if st.button("Generate Bin Labels", type="primary", use_container_width=True):
                    with st.spinner("Generating labels with QR codes..."):
                        # Create progress bar and status container
                        progress_bar = st.progress(0)
                        status_container = st.empty()
                        
                        # Generate the PDF
                        pdf_file = generate_sticker_labels(df, progress_bar, status_container)
                        
                        if pdf_file:
                            st.success("✅ Labels generated successfully!")
                            
                            # Read the PDF file for download
                            with open(pdf_file, "rb") as f:
                                pdf_bytes = f.read()
                            
                            # Create download button
                            st.download_button(
                                label="📥 Download PDF Labels",
                                data=pdf_bytes,
                                file_name=f"{uploaded_file.name.split('.')[0]}_labels.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                            
                            # Clean up temporary file
                            try:
                                os.unlink(pdf_file)
                            except:
                                pass
                        else:
                            st.error("❌ Failed to generate PDF. Please check your data and try again.")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure your file is a valid Excel (.xlsx, .xls) or CSV (.csv) file.")
    
    with col2:
        st.header("📋 Sample Data Format")
        st.markdown("""
        **Example CSV structure:**
        ```
        Part No,Description,Location,QTY/BIN,7M QTY,9M QTY,12M QTY,Store Location
        P001,Engine Filter,A1_B2_C3,5,2,1,2,ST01_R1_S2
        P002,Brake Pad Set,B1_C2_D3,3,1,1,1,ST02_R2_S1
        ```
        
        The system will automatically:
        - Detect part number columns
        - Find description fields  
        - Identify location information
        - Map bus model quantities (7M, 9M, 12M)
        - Generate QR codes with all part info
        """)
        
        st.header("🎯 Label Layout")
        st.markdown("""
        Each label contains:
        - **Part Number** (bold header)
        - **Description** (auto-sized text)
        - **Quantity/Bin** information
        - **Store Location** (7 column grid)
        - **Line Location** (7 column grid)
        - **Bus Model Quantities** (7M, 9M, 12M boxes)
        - **QR Code** with all part information
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🏷️ Professional Bin Label Generator | Built with Streamlit & ReportLab</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
