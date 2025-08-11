import streamlit as st
import pandas as pd
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader
from io import BytesIO
import subprocess
import sys
import re
import tempfile

# Define sticker dimensions - Updated to match template
STICKER_WIDTH = 18 * cm
STICKER_HEIGHT = 8 * cm
STICKER_PAGESIZE = A4  # Changed to A4

# Define content box dimensions
CONTENT_BOX_WIDTH = 18 * cm
CONTENT_BOX_HEIGHT = 8 * cm

# Check for PIL and install if needed
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.write("PIL not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pillow'])
    from PIL import Image as PILImage
    PIL_AVAILABLE = True

# Check for QR code library and install if needed
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    st.write("qrcode not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'qrcode'])
    import qrcode
    QR_AVAILABLE = True

# Define paragraph styles
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, leading=16)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=14, alignment=TA_CENTER, leading=14)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=14, alignment=TA_CENTER, leading=14)

def find_bus_model_column(df_columns):
    """
    Enhanced function to find the bus model column with better detection
    """
    cols = [str(col).upper() for col in df_columns]
    
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
            if pattern(col):
                return df_columns[i]  # Return original column name
    
    return None

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Improved bus model detection that properly matches bus model to MTM box
    Returns a dictionary with keys '7M', '9M', '12M' and their respective quantities
    """
    # Initialize result dictionary
    result = {'7M': '', '9M': '', '12M': ''}
    
    # Get quantity value
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh_raw = row[qty_veh_col]
        if pd.notna(qty_veh_raw):
            if isinstance(qty_veh_raw, float) and qty_veh_raw.is_integer():
                qty_veh = str(int(qty_veh_raw))  # Show as whole number
            else:
                qty_veh = str(qty_veh_raw).strip()  # Keep decimal if present

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
    
    for col in row.index:
        if pd.notna(row[col]):
            col_upper = str(col).upper()
            # High priority columns
            if any(keyword in col_upper for keyword in ['MODEL', 'BUS', 'VEHICLE', 'TYPE']):
                priority_columns.append(col)
            else:
                other_columns.append(col)
    
    # Search priority columns first
    for col in priority_columns:
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
    
    # Method 4: Search in other columns as fallback
    detected_models = []
    for col in other_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            # Use word boundaries to avoid false matches
            if re.search(r'\b7M\b', value_str):
                detected_models.append('7M')
            elif re.search(r'\b9M\b', value_str):
                detected_models.append('9M')
            elif re.search(r'\b12M\b', value_str):
                detected_models.append('12M')
    
    # Remove duplicates while preserving order
    detected_models = list(dict.fromkeys(detected_models))
    
    if detected_models:
        # Use the first detected model
        result[detected_models[0]] = qty_veh
        return result
    
    # Method 5: Last resort - look for standalone numbers that might indicate bus length
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
        import traceback
        traceback.print_exc()
        return None

def extract_store_location_data_from_excel(row_data):
    """Extract store location data from Excel row for Store Location"""
    def get_clean_value(possible_names, default=''):
        """Get clean value from multiple possible column names"""
        for name in possible_names:
            # Try exact match first
            if name in row_data:
                val = row_data[name]
                if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                    return str(val).strip()
            # Try case-insensitive match
            for col in row_data.index:
                if isinstance(col, str) and col.upper() == name.upper():
                    val = row_data[col]
                    if pd.notna(val) and str(val).lower() not in ['nan', 'none', 'null', '']:
                        return str(val).strip()
        return default
    
    # Extract values for store location - Updated based on template
    # The template shows: Mez-c G+1 R 0 1 A 0 1
    # So we need 8 columns for store location
    store_loc_1 = get_clean_value(['Store Loc 1', 'STORE_LOC_1', 'Station Name', 'STATION NAME'], '')
    store_loc_2 = get_clean_value(['Store Loc 2', 'STORE_LOC_2', 'Store Location', 'STORE LOCATION'], '')
    store_loc_3 = get_clean_value(['Store Loc 3', 'STORE_LOC_3', 'ABB ZONE', 'Zone'], '')
    store_loc_4 = get_clean_value(['Store Loc 4', 'STORE_LOC_4', 'ABB LOCATION', 'Location'], '')
    store_loc_5 = get_clean_value(['Store Loc 5', 'STORE_LOC_5', 'ABB FLOOR', 'Floor'], '')
    store_loc_6 = get_clean_value(['Store Loc 6', 'STORE_LOC_6', 'ABB RACK NO', 'Rack'], '')
    store_loc_7 = get_clean_value(['Store Loc 7', 'STORE_LOC_7', 'ABB LEVEL', 'Level'], '')
    store_loc_8 = get_clean_value(['Store Loc 8', 'STORE_LOC_8', 'ABB CELL', 'Cell'], '')
    
    return [store_loc_1, store_loc_2, store_loc_3, store_loc_4, store_loc_5, store_loc_6, store_loc_7, store_loc_8]

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """Generate sticker labels with QR code from Excel data - Updated for A4 and new template"""
    if status_callback:
        status_callback(f"Processing file: {excel_file_path}")
    else:
        st.write(f"Processing file: {excel_file_path}")

    # Create a function to draw the border box around content
    def draw_border(canvas, doc):
        canvas.saveState()
        # Draw border box around the content area - centered on A4 page
        x_offset = (A4[0] - CONTENT_BOX_WIDTH) / 2  # Center horizontally
        y_offset = (A4[1] - CONTENT_BOX_HEIGHT) / 2  # Center vertically
        canvas.setStrokeColor(colors.Color(0, 0, 0, alpha=0.95))
        canvas.setLineWidth(2)
        canvas.rect(
            x_offset,
            y_offset,
            CONTENT_BOX_WIDTH,
            CONTENT_BOX_HEIGHT
        )
        canvas.restoreState()

    # Load the Excel data
    try:
        if excel_file_path.lower().endswith('.csv'):
            df = pd.read_csv(excel_file_path)
        else:
            try:
                df = pd.read_excel(excel_file_path)
            except Exception as e:
                try:
                    df = pd.read_excel(excel_file_path, engine='openpyxl')
                except Exception as e2:
                    df = pd.read_csv(excel_file_path, encoding='latin1')

        if status_callback:
            status_callback(f"Successfully read file with {len(df)} rows")
            status_callback(f"Columns found: {df.columns.tolist()}")
        else:
            st.write(f"Successfully read file with {len(df)} rows")
            st.write("Columns found:", df.columns.tolist())
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
        return None

    # Identify columns (case-insensitive)
    original_columns = df.columns.tolist()
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    # Find relevant columns
    part_no_col = next((col for col in cols if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                   next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))

    desc_col = next((col for col in cols if 'DESC' in col),
                   next((col for col in cols if 'NAME' in col), cols[1] if len(cols) > 1 else part_no_col))

    # Look for "Max capacity" or similar columns instead of "QTY/BIN"
    max_capacity_col = next((col for col in cols if 'MAX' in col and 'CAPACITY' in col), 
                           next((col for col in cols if 'CAPACITY' in col),
                           next((col for col in cols if 'QTY' in col), None)))
    
    # Look for store location column
    store_loc_col = next((col for col in cols if 'STORE' in col and 'LOC' in col),
                      next((col for col in cols if 'STORELOCATION' in col), None))

    # Improved detection of QTY/VEH column
    qty_veh_col = next((col for col in cols if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)

    # Find bus model column using the enhanced detection function
    bus_model_col = find_bus_model_column(original_columns)

    if status_callback:
        status_callback(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Max Capacity: {max_capacity_col}")
        if qty_veh_col:
            status_callback(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            status_callback(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            status_callback(f"Bus Model Column: {bus_model_col}")
    else:
        st.write(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Max Capacity: {max_capacity_col}")
        if qty_veh_col:
            st.write(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            st.write(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            st.write(f"Bus Model Column: {bus_model_col}")

    # Create document with A4 page size and proper margins for centering
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4,
                          topMargin=(A4[1] - CONTENT_BOX_HEIGHT) / 2,
                          bottomMargin=(A4[1] - CONTENT_BOX_HEIGHT) / 2,
                          leftMargin=(A4[0] - CONTENT_BOX_WIDTH) / 2,
                          rightMargin=(A4[0] - CONTENT_BOX_WIDTH) / 2)

    content_width = CONTENT_BOX_WIDTH - 0.4*cm
    all_elements = []

    # Process each row as a single sticker
    total_rows = len(df)
    for index, row in df.iterrows():
        # Update progress
        if status_callback:
            status_callback(f"Creating sticker {index+1} of {total_rows} ({int((index+1)/total_rows*100)}%)")
        
        elements = []

        # Extract data
        part_no = str(row[part_no_col])
        desc = str(row[desc_col])
        
        # Extract Max capacity
        max_capacity = ""
        if max_capacity_col and max_capacity_col in row and pd.notna(row[max_capacity_col]):
            max_capacity = str(row[max_capacity_col])
            
        # Extract QTY/VEH properly
        qty_veh = ""
        if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
            qty_veh = str(row[qty_veh_col])
        
        store_location = str(row[store_loc_col]) if store_loc_col and store_loc_col in row else ""

        # Use enhanced bus model detection
        mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

        # Generate QR code with part information
        qr_data = f"Part No: {part_no}\nDescription: {desc}\nMax Capacity: {max_capacity}\n"
        qr_data += f"Store Location: {store_location}\nQTY/VEH: {qty_veh}"
        
        qr_image = generate_qr_code(qr_data)
        if status_callback and qr_image:
            status_callback(f"QR code generated for part: {part_no}")
        
        # Define row heights - adjusted for larger sticker
        header_row_height = 1.2*cm
        desc_row_height = 1.4*cm
        max_capacity_row_height = 1.0*cm
        store_loc_row_height = 1.2*cm

        # Main table data - Updated to use "Max capacity" instead of "Qty/Bin"
        main_table_data = [
            ["Part No", Paragraph(f"{part_no}", bold_style)],
            ["Description", Paragraph(desc[:60] + "..." if len(desc) > 60 else desc, desc_style)],
            ["Max capacity", Paragraph(str(max_capacity), qty_style)]
        ]

        # Create main table
        main_table = Table(main_table_data,
                         colWidths=[content_width/3, content_width*2/3],
                         rowHeights=[header_row_height, desc_row_height, max_capacity_row_height])

        main_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 14),
        ]))

        elements.append(main_table)

        # Store Location section - Updated for 8 columns as per template
        store_loc_label = Paragraph("Store Location", ParagraphStyle(
            name='StoreLoc', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER
        ))
        
        # Total width for the 8 inner columns
        inner_table_width = content_width * 2 / 3

        # Define proportional widths for 8 columns
        col_proportions = [1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        total_proportion = sum(col_proportions)

        # Calculate column widths based on proportions 
        inner_col_widths = [w * inner_table_width / total_proportion for w in col_proportions]

        # Extract store location values from Excel data
        store_loc_values = extract_store_location_data_from_excel(row)

        store_loc_inner_table = Table(
            [store_loc_values],
            colWidths=inner_col_widths,
            rowHeights=[store_loc_row_height]
        )
        store_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
        ]))
        
        store_loc_table = Table(
            [[store_loc_label, store_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[store_loc_row_height]
        )
        store_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(store_loc_table)

        # Add spacer before bottom section
        elements.append(Spacer(1, 0.5*cm))

        # Bottom section - MTM boxes and QR code
        mtm_box_width = 2*cm
        mtm_row_height = 2*cm

        # Create MTM boxes with detected quantities
        position_matrix_data = [
            ["7M", "9M", "12M"],
            [
                Paragraph(f"<b>{mtm_quantities['7M']}</b>", ParagraphStyle(
                    name='Bold7M', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER
                )) if mtm_quantities['7M'] else "",
                Paragraph(f"<b>{mtm_quantities['9M']}</b>", ParagraphStyle(
                    name='Bold9M', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER
                )) if mtm_quantities['9M'] else "",
                Paragraph(f"<b>{mtm_quantities['12M']}</b>", ParagraphStyle(
                    name='Bold12M', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER
                )) if mtm_quantities['12M'] else ""
            ]
        ]

        mtm_table = Table(
            position_matrix_data,
            colWidths=[mtm_box_width, mtm_box_width, mtm_box_width],
            rowHeights=[mtm_row_height/2, mtm_row_height/2]
        )

        mtm_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
        ]))

        # QR code - larger size for bigger sticker
        qr_width = 3*cm
        qr_height = 3*cm

        if qr_image:
            # Update QR image size
            qr_image.drawWidth = qr_width
            qr_image.drawHeight = qr_height
            qr_table = Table(
                [[qr_image]],
                colWidths=[qr_width],
                rowHeights=[qr_height]
            )
        else:
            qr_table = Table(
                [[Paragraph("QR", ParagraphStyle(
                    name='QRPlaceholder', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER
                ))]],
                colWidths=[qr_width],
                rowHeights=[qr_height]
            )

        qr_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Calculate spacing for bottom row
        total_mtm_width = 3 * mtm_box_width
        spacer_width = content_width - total_mtm_width - qr_width

        # Combine MTM boxes and QR code in one row
        bottom_row = Table(
            [[mtm_table, "", qr_table]],
            colWidths=[total_mtm_width, spacer_width, qr_width],
            rowHeights=[max(mtm_row_height, qr_height)]
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
        if status_callback:
            status_callback(f"PDF generated successfully: {output_pdf_path}")
        else:
            st.success(f"PDF generated successfully: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        error_msg = f"Error building PDF: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
            import traceback
            traceback.print_exc()
        return None

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Bin Label Generator", page_icon="🏷️", layout="wide")
    
    st.title("🏷️ Bin Label Generator")
    st.markdown(
        "<p style='font-size:18px; font-style:italic; margin-top:-10px; text-align:left;'>"
        "Designed and Developed by Agilomatrix</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your Excel or CSV file containing part information"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_input_path = tmp_file.name
        
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Preview data
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                preview_df = pd.read_csv(temp_input_path).head(5)
            else:
                preview_df = pd.read_excel(temp_input_path).head(5)
            
            st.subheader("📊 Data Preview (First 5 rows)")
            st.dataframe(preview_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error previewing file: {e}")
            return
        
        # Column mapping section
        st.subheader("🔧 Column Detection")
        
        # Auto-detect columns and show them
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df_full = pd.read_csv(temp_input_path)
            else:
                df_full = pd.read_excel(temp_input_path)
            
            # Show detected columns
            cols_upper = [col.upper() if isinstance(col, str) else col for col in df_full.columns]
            
            # Auto-detect key columns
            part_no_col = next((col for col in cols_upper if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                             next((col for col in cols_upper if col in ['PARTNO', 'PART']), cols_upper[0] if cols_upper else ''))
            
            desc_col = next((col for col in cols_upper if 'DESC' in col),
                           next((col for col in cols_upper if 'NAME' in col), cols_upper[1] if len(cols_upper) > 1 else ''))
            
            # Look for Max capacity column instead of Qty/Bin
            max_capacity_col = next((col for col in cols_upper if 'MAX' in col and 'CAPACITY' in col), 
                                   next((col for col in cols_upper if 'CAPACITY' in col),
                                   next((col for col in cols_upper if 'QTY' in col), '')))
            
            qty_veh_col = next((col for col in cols_upper if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH'])), '')
            
            bus_model_col = find_bus_model_column(df_full.columns.tolist())
            
            # Display detected columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Part Number Column:** {part_no_col}")
                st.info(f"**Description Column:** {desc_col}")
                st.info(f"**Max Capacity Column:** {max_capacity_col}")
                
            with col2:
                st.info(f"**Qty/Vehicle Column:** {qty_veh_col if qty_veh_col else 'Not detected'}")
                st.info(f"**Bus Model Column:** {bus_model_col if bus_model_col else 'Not detected'}")
                st.info(f"**Store Location:** Available in multiple columns")
            
        except Exception as e:
            st.error(f"Error analyzing columns: {e}")
            return
        
        # Generate labels section
        st.subheader("🚀 Generate Labels")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🏷️ Generate PDF Labels", type="primary", use_container_width=True):
                # Create progress container
                progress_container = st.empty()
                status_container = st.empty()
                
                # Create temporary output file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_output:
                    temp_output_path = tmp_output.name
                
                # Progress tracking
                def update_status(message):
                    status_container.info(f"📊 {message}")
                
                try:
                    # Generate the PDF
                    update_status("Starting label generation...")
                    
                    result_path = generate_sticker_labels(
                        temp_input_path, 
                        temp_output_path,
                        status_callback=update_status
                    )
                    
                    if result_path:
                        # Success - provide download
                        with open(result_path, 'rb') as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        status_container.success("✅ Labels generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Labels",
                            data=pdf_data,
                            file_name=f"sticker_labels_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # Show file size
                        file_size = len(pdf_data) / 1024  # KB
                        st.info(f"📄 PDF size: {file_size:.1f} KB | Pages: {len(df_full)}")
                        
                    else:
                        status_container.error("❌ Failed to generate labels")
                        
                except Exception as e:
                    status_container.error(f"❌ Error: {str(e)}")
                    st.exception(e)
                
                finally:
                    # Cleanup temporary files
                    try:
                        if os.path.exists(temp_input_path):
                            os.unlink(temp_input_path)
                        if os.path.exists(temp_output_path):
                            os.unlink(temp_output_path)
                    except:
                        pass
        
        with col2:
            if st.button("🔍 Preview Sample", use_container_width=True):
                st.info("Preview functionality - shows first label design")
                # You could add preview functionality here
        
        # Additional information
        st.subheader("ℹ️ Label Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("""
            **Label Features:**
            - 📏 A4 page with 18cm x 8cm stickers
            - 🔢 QR code for each part
            - 📍 Store location tracking (8 fields)
            - 🚌 Bus model detection (7M, 9M, 12M)
            - 📦 Max capacity field
            """)
        
        with info_col2:
            st.markdown("""
            **Supported Columns:**
            - Part Number/Part No
            - Description/Name
            - Max Capacity/Capacity
            - Qty/Veh, Qty per Vehicle
            - Bus Model/Vehicle Type
            - Store Location (multiple fields)
            """)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an Excel or CSV file to get started")
        
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload your file** - Excel (.xlsx, .xls) or CSV format
        2. **Review data preview** - Check if your data looks correct
        3. **Verify column detection** - Ensure columns are properly identified
        4. **Generate labels** - Click the button to create your PDF
        5. **Download** - Get your professional sticker labels
        """)
        
        st.subheader("💡 Tips")
        st.markdown("""
        - Use clear column headers like "Part No", "Description", "Max Capacity"
        - For bus models, use "7M", "9M", "12M" format
        - Include quantity information in "Qty/Veh" columns
        - Store location will be distributed across 8 fields automatically
        """)
        
        # Sample data format
        st.subheader("📊 Sample Data Format")
        sample_data = pd.DataFrame({
            'Part No': ['AA0107020000', 'DEF456', 'GHI789'],
            'Description': ['REAR SUSPENSION-WIL', 'Brake Pad Set', 'Oil Filter'],
            'Max Capacity': [9, 10, 8],
            'Qty/Veh': [2, 4, 1],
            'Bus Model': ['12M', '9M', '7M'],
            'Store Loc 1': ['Mez-c', 'G+1', 'B1'],
            'Store Loc 2': ['G+1', 'R2', 'L3'],
            'Store Loc 3': ['R', 'A', 'B']
        })
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()
