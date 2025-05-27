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
import re
import tempfile
import base64

# Install required packages if not available
try:
    from PIL import Image as PILImage
except ImportError:
    st.error("PIL not available. Please install: pip install pillow")
    st.stop()

try:
    import qrcode
except ImportError:
    st.error("qrcode not available. Please install: pip install qrcode")
    st.stop()

# Define sticker dimensions
STICKER_WIDTH = 10 * cm
STICKER_HEIGHT = 15 * cm
STICKER_PAGESIZE = (STICKER_WIDTH, STICKER_HEIGHT)

# Define content box dimensions
CONTENT_BOX_WIDTH = 10 * cm  # Same width as page
CONTENT_BOX_HEIGHT = 7.2 * cm  # Half the page height

# Define paragraph styles
bold_style = ParagraphStyle(
    name='Bold', 
    fontName='Helvetica-Bold', 
    fontSize=16, 
    alignment=TA_CENTER, 
    leading=14
)
desc_style = ParagraphStyle(
    name='Description', 
    fontName='Helvetica', 
    fontSize=11, 
    alignment=TA_CENTER, 
    leading=12
)
qty_style = ParagraphStyle(
    name='Quantity', 
    fontName='Helvetica', 
    fontSize=11, 
    alignment=TA_CENTER, 
    leading=12
)

def find_bus_model_column(df_columns):
    """
    Enhanced function to find the bus model column with better detection
    """
    if not df_columns:
        return None
        
    cols = [str(col).upper() if col is not None else '' for col in df_columns]
    
    # Priority order for bus model column detection
    patterns = [
        # Exact matches (highest priority)
        lambda col: col == 'BUS_MODEL',
        lambda col: col == 'BUSMODEL',
        lambda col: col == 'BUS MODEL',
        lambda col: col == 'MODEL',
        lambda col: col == 'BUS_TYPE',
        lambda col: col == 'BUSTYPE',
        lambda col: col == 'BUS TYPE',
        lambda col: col == 'VEHICLE_TYPE',
        lambda col: col == 'VEHICLETYPE',
        lambda col: col == 'VEHICLE TYPE',
        # Partial matches (lower priority)
        lambda col: 'BUS' in col and 'MODEL' in col,
        lambda col: 'BUS' in col and 'TYPE' in col,
        lambda col: 'VEHICLE' in col and 'MODEL' in col,
        lambda col: 'VEHICLE' in col and 'TYPE' in col,
        lambda col: 'MODEL' in col,
        lambda col: 'BUS' in col,
        lambda col: 'VEHICLE' in col,
    ]
    
    for pattern in patterns:
        for i, col in enumerate(cols):
            try:
                if pattern(col):
                    return df_columns[i]  # Return original column name
            except:
                continue
    
    return None

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Improved bus model detection that properly matches bus model to MTM box
    Returns a dictionary with keys '7M', '9M', '12M' and their respective quantities
    """
    # Initialize result dictionary
    result = {'7M': '', '9M': '', '12M': ''}
    
    if row is None:
        return result
    
    # Get quantity value
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = str(row[qty_veh_col]).strip()
    
    if not qty_veh:
        return result
    
    # Method 1: Check if quantity already contains model info (e.g., "9M:2", "7M-3", "12M 5")
    qty_pattern = r'(\d+M)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    
    if matches:
        # If we found model-quantity pairs in the qty_veh field itself
        for model, quantity in matches:
            if model in result:
                result[model] = quantity
        return result
    
    # Method 2: Look for bus model in dedicated bus model column first
    detected_model = None
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        bus_model_value = str(row[bus_model_col]).strip().upper()
        
        # Check for exact matches first
        if bus_model_value in ['7M', '7']:
            detected_model = '7M'
        elif bus_model_value in ['9M', '9']:
            detected_model = '9M'
        elif bus_model_value in ['12M', '12']:
            detected_model = '12M'
        # Check for patterns within the text
        elif re.search(r'\b7M\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9M\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12M\b', bus_model_value):
            detected_model = '12M'
        # Check for standalone numbers
        elif re.search(r'\b7\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12\b', bus_model_value):
            detected_model = '12M'
    
    # If we found a model in the dedicated column, use it
    if detected_model:
        result[detected_model] = qty_veh
        return result
    
    # Method 3: Search through all columns systematically with priority
    # First, search in columns that are most likely to contain bus model info
    priority_columns = []
    other_columns = []
    
    try:
        for col in row.index:
            if pd.notna(row[col]):
                col_upper = str(col).upper()
                # High priority columns
                if any(keyword in col_upper for keyword in ['MODEL', 'BUS', 'VEHICLE', 'TYPE']):
                    priority_columns.append(col)
                else:
                    other_columns.append(col)
    except:
        return result
    
    # Search priority columns first
    for col in priority_columns:
        try:
            if pd.notna(row[col]):
                value_str = str(row[col]).upper()
                
                # Look for exact matches first
                if re.search(r'\b7M\b', value_str):
                    result['7M'] = qty_veh
                    return result
                elif re.search(r'\b9M\b', value_str):
                    result['9M'] = qty_veh
                    return result
                elif re.search(r'\b12M\b', value_str):
                    result['12M'] = qty_veh
                    return result
                # Then look for standalone numbers in context
                elif re.search(r'\b7\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                    result['7M'] = qty_veh
                    return result
                elif re.search(r'\b9\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                    result['9M'] = qty_veh
                    return result
                elif re.search(r'\b12\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                    result['12M'] = qty_veh
                    return result
        except:
            continue
    
    # Method 4: Search in other columns as fallback
    detected_models = []
    for col in other_columns:
        try:
            if pd.notna(row[col]):
                value_str = str(row[col]).upper()
                
                # Use word boundaries to avoid false matches
                if re.search(r'\b7M\b', value_str):
                    detected_models.append('7M')
                elif re.search(r'\b9M\b', value_str):
                    detected_models.append('9M')
                elif re.search(r'\b12M\b', value_str):
                    detected_models.append('12M')
        except:
            continue
    
    # Remove duplicates while preserving order
    detected_models = list(dict.fromkeys(detected_models))
    
    if detected_models:
        # Use the first detected model
        result[detected_models[0]] = qty_veh
        return result
    
    # Method 5: Last resort - look for standalone numbers that might indicate bus length
    try:
        for col in row.index:
            if pd.notna(row[col]):
                value_str = str(row[col]).strip()
                
                # Look for exact matches of just the number
                if value_str == '7':
                    result['7M'] = qty_veh
                    return result
                elif value_str == '9':
                    result['9M'] = qty_veh
                    return result
                elif value_str == '12':
                    result['12M'] = qty_veh
                    return result
    except:
        pass
    
    # Method 6: If still no model detected, return empty (no boxes filled)
    return result

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

def safe_str(value):
    """Safely convert value to string, handling NaN and None"""
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()

def find_column(df_columns, patterns):
    """Find column matching any of the given patterns"""
    if not df_columns:
        return None
        
    cols_upper = [str(col).upper() if col is not None else '' for col in df_columns]
    
    for pattern in patterns:
        for i, col in enumerate(cols_upper):
            if pattern in col:
                return df_columns[i]
    return None

def generate_sticker_labels(df, progress_bar=None, status_container=None):
    """Generate sticker labels with QR code from DataFrame"""
    
    try:
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

        # Validate DataFrame
        if df is None or df.empty:
            if status_container:
                status_container.error("DataFrame is empty or None")
            return None

        # Identify columns (case-insensitive)
        original_columns = df.columns.tolist()
        df_upper = df.copy()
        df_upper.columns = [str(col).upper() if col is not None else f'COL_{i}' for i, col in enumerate(df_upper.columns)]
        cols = df_upper.columns.tolist()

        # Find relevant columns with better error handling
        part_no_col = find_column(cols, ['PART_NO', 'PARTNO', 'PART NO', 'PART']) or cols[0]
        desc_col = find_column(cols, ['DESC', 'DESCRIPTION', 'NAME']) or (cols[1] if len(cols) > 1 else part_no_col)
        
        # Look specifically for "QTY/BIN" column first, then fall back to general QTY column
        qty_bin_col = find_column(cols, ['QTY/BIN', 'QTY_BIN', 'QTYBIN', 'QTY PER BIN']) or find_column(cols, ['QTY', 'QUANTITY'])
        
        loc_col = find_column(cols, ['LOC', 'LOCATION', 'POS', 'POSITION']) or (cols[2] if len(cols) > 2 else desc_col)

        # Improved detection of QTY/VEH column
        qty_veh_col = find_column(cols, ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])

        # Look for store location column
        store_loc_col = find_column(cols, ['STORE_LOCATION', 'STORELOCATION', 'STORE LOC'])

        # Find bus model column using the enhanced detection function
        bus_model_col = find_bus_model_column(original_columns)

        if status_container:
            status_container.write(f"**Using columns:**")
            status_container.write(f"- Part No: {part_no_col}")
            status_container.write(f"- Description: {desc_col}")
            status_container.write(f"- Location: {loc_col}")
            if qty_bin_col:
                status_container.write(f"- Qty/Bin: {qty_bin_col}")
            if qty_veh_col:
                status_container.write(f"- Qty/Veh Column: {qty_veh_col}")
            if store_loc_col:
                status_container.write(f"- Store Location Column: {store_loc_col}")
            if bus_model_col:
                status_container.write(f"- Bus Model Column: {bus_model_col}")

        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            output_pdf_path = tmp_file.name

        # Create document with minimal margins
        doc = SimpleDocTemplate(
            output_pdf_path, 
            pagesize=STICKER_PAGESIZE,
            topMargin=0.2*cm,  # Minimal top margin
            bottomMargin=(STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm),  # Adjust bottom margin accordingly
            leftMargin=0.1*cm, 
            rightMargin=0.1*cm
        )

        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        all_elements = []

        # Process each row as a single sticker
        total_rows = len(df_upper)
        for index, row in df_upper.iterrows():
            try:
                # Update progress
                if progress_bar:
                    progress_bar.progress((index + 1) / total_rows)
                if status_container:
                    status_container.write(f"Creating sticker {index+1} of {total_rows}")
                
                elements = []

                # Extract data with safe string conversion
                part_no = safe_str(row.get(part_no_col, ''))
                desc = safe_str(row.get(desc_col, ''))
                
                # Extract QTY/BIN properly
                qty_bin = safe_str(row.get(qty_bin_col, '')) if qty_bin_col else ""
                    
                # Extract QTY/VEH properly
                qty_veh = safe_str(row.get(qty_veh_col, '')) if qty_veh_col else ""
                
                location_str = safe_str(row.get(loc_col, '')) if loc_col else ""
                store_location = safe_str(row.get(store_loc_col, '')) if store_loc_col else ""
                location_parts = parse_location_string(location_str)

                # Use enhanced bus model detection
                mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

                # Generate QR code with part information
                qr_data = f"Part No: {part_no}\nDescription: {desc}\nLocation: {location_str}\n"
                qr_data += f"Store Location: {store_location}\nQTY/VEH: {qty_veh}\nQTY/BIN: {qty_bin}"
                
                qr_image = generate_qr_code(qr_data)
                
                # Define row heights
                header_row_height = 0.9*cm
                desc_row_height = 1.0*cm
                qty_row_height = 0.5*cm
                location_row_height = 0.5*cm

                # Main table data
                main_table_data = [
                    ["Part No", Paragraph(f"{part_no}", bold_style)],
                    ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, desc_style)],
                    ["Qty/Bin", Paragraph(qty_bin, qty_style)]
                ]

                # Create main table
                main_table = Table(
                    main_table_data,
                    colWidths=[content_width/3, content_width*2/3],
                    rowHeights=[header_row_height, desc_row_height, qty_row_height]
                )

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
                col_proportions = [1.5, 2, 0.7, 0.8, 1, 1, 0.9]
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

                # Bottom section - Enhanced with intelligent bus model detection
                mtm_box_width = 1.2*cm
                mtm_row_height = 1.5*cm

                # Create MTM boxes with detected quantities
                position_matrix_data = [
                    ["7M", "9M", "12M"],
                    [
                        Paragraph(f"<b>{mtm_quantities['7M']}</b>", ParagraphStyle(
                            name='Bold7M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                        )) if mtm_quantities['7M'] else "",
                        Paragraph(f"<b>{mtm_quantities['9M']}</b>", ParagraphStyle(
                            name='Bold9M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                        )) if mtm_quantities['9M'] else "",
                        Paragraph(f"<b>{mtm_quantities['12M']}</b>", ParagraphStyle(
                            name='Bold12M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                        )) if mtm_quantities['12M'] else ""
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
                if index < len(df_upper) - 1:
                    all_elements.append(PageBreak())
                    
            except Exception as e:
                if status_container:
                    status_container.error(f"Error processing row {index+1}: {str(e)}")
                continue

        # Build the document
        try:
            # Pass the draw_border function to build to add border box
            doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
            if status_container:
                status_container.write("✅ PDF generated successfully!")
            return output_pdf_path
        except Exception as e:
            if status_container:
                status_container.error(f"Error building PDF: {e}")
            # Clean up file if creation failed
            try:
                os.unlink(output_pdf_path)
            except:
                pass
            return None
            
    except Exception as e:
        if status_container:
            status_container.error(f"Error in generate_sticker_labels: {str(e)}")
        return None

def get_download_link(file_path, filename):
    """Generate a download link for the PDF file"""
    try:
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 Download PDF</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Sticker Label Generator",
        page_icon="🏷️",
        layout="wide"
    )
    
    st.title("🏷️ Sticker Label Generator")
    st.markdown("Upload your Excel or CSV file to generate professional sticker labels with QR codes.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your data file containing part information"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
            # Generate button
            if st.button("🔄 Generate Sticker Labels", type="primary", use_container_width=True):
                # Create containers for progress and status
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                try:
                    with st.spinner("Generating sticker labels..."):
                        # Generate the PDF
                        pdf_path = generate_sticker_labels(
                            df, 
                            progress_bar=progress_bar, 
                            status_container=status_container
                        )
                    if pdf_path:
                            progress_bar.progress(1.0)
                            status_container.success("🎉 Labels generated successfully!")
                            
                            # Create download button
                            filename = f"sticker_labels_{uploaded_file.name.split('.')[0]}.pdf"
                            download_link = get_download_link(pdf_path, filename)
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                            # Show file info
                            st.info(f"📄 Generated PDF: {filename}")
                            
                            # Clean up temporary file after a delay
                            import time
                            time.sleep(1)
                            try:
                                os.unlink(pdf_path)
                            except:
                                pass  # Ignore cleanup errors
                        else:
                            status_container.error("❌ Failed to generate PDF. Please check your data and try again.")
                            
                except Exception as e:
                    st.error(f"❌ Error generating labels: {str(e)}")
                    if 'pdf_path' in locals() and pdf_path:
                        try:
                            os.unlink(pdf_path)
                        except:
                            pass
                finally:
                    # Clear progress bar
                    progress_bar.empty()
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Please make sure your file is a valid Excel (.xlsx, .xls) or CSV (.csv) file.")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload an Excel or CSV file to get started.")
        
        with st.expander("📋 File Format Requirements", expanded=False):
            st.markdown("""
            **Your file should contain the following columns:**
            - **Part Number** (Part No, PartNo, Part_No, etc.)
            - **Description** (Description, Desc, Name, etc.)
            - **Location** (Location, Loc, Position, etc.)
            - **Quantity per Bin** (Qty/Bin, QtyBin, Qty_Bin, etc.)
            - **Quantity per Vehicle** (Qty/Veh, QtyVeh, Qty_Veh, etc.) - Optional
            - **Bus Model** (Bus_Model, BusModel, Model, etc.) - Optional
            - **Store Location** (Store_Location, StoreLocation, etc.) - Optional
            
            **Supported Bus Models:** 7M, 9M, 12M
            
            **Note:** Column names are case-insensitive and the system will automatically detect similar column names.
            """)
        
        with st.expander("🎯 Features", expanded=False):
            st.markdown("""
            **This tool generates professional sticker labels with:**
            - 📦 Part number and description
            - 📍 Location information (Store Location + Line Location)
            - 📊 Quantity per bin information
            - 🚌 Bus model detection (7M, 9M, 12M) with quantities
            - 📱 QR codes containing all part information
            - 🖨️ Print-ready PDF format (10cm x 15cm stickers)
            - 📋 Professional layout with borders and clear sections
            """)

if __name__ == "__main__":
    main()
