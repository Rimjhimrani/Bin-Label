import streamlit as st

# This must be the very first Streamlit command
st.set_page_config(
    page_title="Bin Label Generator",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
import re

# Define sticker dimensions
STICKER_WIDTH = 10 * cm
STICKER_HEIGHT = 15 * cm
STICKER_PAGESIZE = (STICKER_WIDTH, STICKER_HEIGHT)

# Define content box dimensions
CONTENT_BOX_WIDTH = 10 * cm  # Same width as page
CONTENT_BOX_HEIGHT = 7.2 * cm  # Half the page height

# Function to install packages if needed
def install_if_needed(package_name, import_name=None):
    """Install package if not available"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        try:
            st.warning(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            return True
        except Exception as e:
            st.error(f"Failed to install {package_name}: {e}")
            return False

# Check and install required packages
def check_dependencies():
    """Check and install required packages"""
    packages = [
        ('pillow', 'PIL'),
        ('qrcode', 'qrcode'),
        ('reportlab', 'reportlab'),
        ('openpyxl', 'openpyxl')
    ]
    
    all_installed = True
    for package, import_name in packages:
        if not install_if_needed(package, import_name):
            all_installed = False
    
    return all_installed

# Import PIL and qrcode after ensuring they're installed
PIL_AVAILABLE = False
QR_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    pass

try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    pass

# Define paragraph styles
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

def generate_qr_code(data_string):
    """
    Generate a QR code from the given data string
    """
    if not QR_AVAILABLE:
        return None
        
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

def parse_location_string(location_str):
    """Parse a location string into components for table display"""
    # Initialize with empty values
    location_parts = [''] * 7

    if not location_str or not isinstance(location_str, str):
        return location_parts

    # Remove any extra spaces
    location_str = location_str.strip()

    # Try to parse location components
    import re
    pattern = r'([^_\s]+)'
    matches = re.findall(pattern, location_str)

    # Fill the available parts
    for i, match in enumerate(matches[:7]):
        location_parts[i] = match

    return location_parts

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

def debug_detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Debug version that prints what it finds in each step
    This helps troubleshoot bus model detection issues
    """
    print(f"\n=== DEBUG: Bus Model Detection ===")
    print(f"Quantity column: {qty_veh_col}")
    print(f"Bus model column: {bus_model_col}")
    
    # Print all row data for debugging
    print("\nAll row data:")
    for col, val in row.items():
        if pd.notna(val):
            print(f"  {col}: '{val}' (type: {type(val)})")
    
    # Get quantity value
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = str(row[qty_veh_col]).strip()
    print(f"\nQuantity value: '{qty_veh}'")
    
    # Check for embedded model info in quantity
    qty_pattern = r'(\d+M)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    if matches:
        print(f"Found embedded model-quantity pairs: {matches}")
    
    # Check dedicated bus model column
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        bus_model_value = str(row[bus_model_col]).strip().upper()
        print(f"Bus model column value: '{bus_model_value}'")
    
    # Search for models in all columns
    print(f"\nSearching for bus models in all columns:")
    for col, val in row.items():
        if pd.notna(val):
            val_str = str(val).upper()
            models_found = []
            if re.search(r'\b7M\b', val_str):
                models_found.append('7M')
            if re.search(r'\b9M\b', val_str):
                models_found.append('9M')
            if re.search(r'\b12M\b', val_str):
                models_found.append('12M')
            if models_found:
                print(f"  {col}: Found models {models_found}")
    
    result = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)
    print(f"\nFinal result: {result}")
    print("=== END DEBUG ===\n")
    
    return result

def generate_sticker_labels(excel_file_path, output_pdf_path, progress_bar=None, status_placeholder=None):
    """Generate sticker labels with QR code from Excel data"""
    if status_placeholder:
        status_placeholder.info(f"Processing file: {excel_file_path}")

    # Create a function to draw the border box around content
    def draw_border(canvas, doc):
        canvas.saveState()
        # Draw border box around the content area (10cm x 7.5cm)
        # Position it at the top of the page with minimal margin
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm  # Position at top with minimal margin
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1.6)  # Increased line width by ~7% (from 1.5 to 1.6)
        canvas.rect(
            x_offset + doc.leftMargin,
            y_offset,
            CONTENT_BOX_WIDTH - 0.2*cm,  # Account for margins
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
                    df = pd.read_csv(excel_file_path, encoding='df = pd.read_csv(excel_file_path, encoding='utf-8')
        
        if status_placeholder:
            status_placeholder.info(f"Data loaded successfully. Found {len(df)} rows.")
        
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        if status_placeholder:
            status_placeholder.error(error_msg)
        st.error(error_msg)
        return False
    
    if df.empty:
        error_msg = "The uploaded file is empty or contains no valid data."
        if status_placeholder:
            status_placeholder.error(error_msg)
        st.error(error_msg)
        return False
    
    # Find required columns
    if status_placeholder:
        status_placeholder.info("Analyzing column structure...")
    
    # Find Part Number column
    part_no_col = None
    for col in df.columns:
        if any(keyword in str(col).upper() for keyword in ['PART', 'PART_NO', 'PARTNO', 'PART NUMBER']):
            part_no_col = col
            break
    
    # Find Description column
    desc_col = None
    for col in df.columns:
        if any(keyword in str(col).upper() for keyword in ['DESC', 'DESCRIPTION', 'ITEM']):
            desc_col = col
            break
    
    # Find Location column
    location_col = None
    for col in df.columns:
        if any(keyword in str(col).upper() for keyword in ['LOCATION', 'LOC', 'POSITION', 'BIN']):
            location_col = col
            break
    
    # Find Quantity/Vehicle column
    qty_veh_col = None
    for col in df.columns:
        if any(keyword in str(col).upper() for keyword in ['QTY', 'QUANTITY', 'VEH', 'VEHICLE', 'COUNT']):
            qty_veh_col = col
            break
    
    # Find Bus Model column using enhanced detection
    bus_model_col = find_bus_model_column(df.columns)
    
    if status_placeholder:
        status_placeholder.info(f"Columns detected - Part: {part_no_col}, Desc: {desc_col}, Location: {location_col}, Qty: {qty_veh_col}, Bus Model: {bus_model_col}")
    
    # Create PDF document with custom page size
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=STICKER_PAGESIZE,
        rightMargin=0.5*cm,
        leftMargin=0.5*cm,
        topMargin=0.5*cm,
        bottomMargin=0.5*cm
    )
    
    # Story to hold all elements
    story = []
    
    total_rows = len(df)
    processed_rows = 0
    
    for index, row in df.iterrows():
        if status_placeholder:
            status_placeholder.info(f"Processing row {processed_rows + 1} of {total_rows}")
        
        if progress_bar:
            progress_bar.progress((processed_rows + 1) / total_rows)
        
        # Extract data from row
        part_no = str(row[part_no_col]) if part_no_col and pd.notna(row[part_no_col]) else ""
        description = str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else ""
        location = str(row[location_col]) if location_col and pd.notna(row[location_col]) else ""
        
        # Parse location string
        location_parts = parse_location_string(location)
        
        # Detect bus model and quantity
        bus_qty_dict = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)
        
        # Create QR code data
        qr_data = f"Part: {part_no}\nDesc: {description}\nLoc: {location}"
        qr_code_img = generate_qr_code(qr_data) if QR_AVAILABLE else None
        
        # Create the main table structure
        table_data = []
        
        # Header row with Part Number
        table_data.append([Paragraph(f"<b>{part_no}</b>", bold_style)])
        
        # Description row
        if description:
            table_data.append([Paragraph(description, desc_style)])
        
        # Location table
        location_headers = ["Building", "Floor", "Section", "Row", "Column", "Shelf", "Position"]
        location_table_data = [location_headers, location_parts]
        
        location_table = Table(location_table_data, colWidths=[1.4*cm] * 7)
        location_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # MTM box table
        mtm_headers = ["7M", "9M", "12M"]
        mtm_values = [bus_qty_dict.get('7M', ''), bus_qty_dict.get('9M', ''), bus_qty_dict.get('12M', '')]
        mtm_table_data = [mtm_headers, mtm_values]
        
        mtm_table = Table(mtm_table_data, colWidths=[3.2*cm] * 3, rowHeights=[0.8*cm, 1.2*cm])
        mtm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1.2, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Create layout with QR code if available
        if qr_code_img:
            # Layout with QR code on the right
            content_table_data = [
                [location_table, qr_code_img],
                [mtm_table, ""]
            ]
            content_table = Table(content_table_data, colWidths=[7*cm, 2.5*cm])
            content_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ]))
        else:
            # Layout without QR code
            content_table_data = [
                [location_table],
                [mtm_table]
            ]
            content_table = Table(content_table_data, colWidths=[9.5*cm])
            content_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
        
        # Add all elements to story
        for row_data in table_data:
            story.append(Table([row_data], colWidths=[9.5*cm], rowHeights=[0.8*cm]))
            story.append(Spacer(1, 0.1*cm))
        
        story.append(content_table)
        
        # Add page break if not the last item
        if index < len(df) - 1:
            story.append(PageBreak())
        
        processed_rows += 1
    
    # Build PDF
    if status_placeholder:
        status_placeholder.info("Generating PDF...")
    
    try:
        doc.build(story, onFirstPage=draw_border, onLaterPages=draw_border)
        if status_placeholder:
            status_placeholder.success(f"PDF generated successfully: {output_pdf_path}")
        return True
    except Exception as e:
        error_msg = f"Error generating PDF: {str(e)}"
        if status_placeholder:
            status_placeholder.error(error_msg)
        st.error(error_msg)
        return False

# Streamlit UI
def main():
    st.title("🏷️ Bin Label Generator")
    st.markdown("Generate professional bin labels with QR codes from Excel/CSV data")
    
    # Check dependencies
    if not check_dependencies():
        st.error("Failed to install required dependencies. Please check your environment.")
        return
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Excel or CSV file",
            type=['xlsx', 'xls', 'csv'],
            help="Upload your data file containing part numbers, descriptions, locations, and quantities"
        )
        
        if uploaded_file:
            st.success("File uploaded successfully!")
            
            # Preview data
            if st.checkbox("Preview data"):
                try:
                    if uploaded_file.name.lower().endswith('.csv'):
                        preview_df = pd.read_csv(uploaded_file)
                    else:
                        preview_df = pd.read_excel(uploaded_file)
                    
                    st.subheader("Data Preview")
                    st.dataframe(preview_df.head(10))
                    st.info(f"Total rows: {len(preview_df)}")
                    
                    # Show column information
                    st.subheader("Column Information")
                    for col in preview_df.columns:
                        st.write(f"• {col}")
                        
                except Exception as e:
                    st.error(f"Error previewing file: {e}")
    
    # Main content area
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📋 Generate Labels")
            
            if st.button("Generate PDF Labels", type="primary", use_container_width=True):
                # Create temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx' if uploaded_file.name.endswith('.xlsx') else '.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Create output file path
                output_pdf_path = tempfile.mktemp(suffix='.pdf')
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                # Generate labels
                success = generate_sticker_labels(
                    tmp_file_path, 
                    output_pdf_path, 
                    progress_bar=progress_bar,
                    status_placeholder=status_placeholder
                )
                
                if success:
                    # Provide download button
                    with open(output_pdf_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                        
                    st.download_button(
                        label="📥 Download PDF Labels",
                        data=pdf_data,
                        file_name=f"bin_labels_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ Labels generated successfully!")
                
                # Cleanup temporary files
                try:
                    os.unlink(tmp_file_path)
                    if os.path.exists(output_pdf_path):
                        os.unlink(output_pdf_path)
                except:
                    pass
        
        with col2:
            st.header("ℹ️ Instructions")
            st.markdown("""
            **Required Columns:**
            - Part Number/Part No
            - Description/Desc
            - Location/Loc
            - Quantity/Qty/Vehicle/Veh
            
            **Optional Columns:**
            - Bus Model/Model (7M, 9M, 12M)
            
            **Location Format:**
            Use underscore or space separated values:
            `Building_Floor_Section_Row_Column_Shelf_Position`
            
            **Bus Model Detection:**
            - Automatic detection from quantity or model columns
            - Supports formats like "7M", "9M", "12M"
            - Can detect embedded info like "9M:2" in quantity
            
            **Output:**
            - Professional labels with QR codes
            - 10cm x 15cm sticker size
            - Location breakdown table
            - MTM (7M/9M/12M) quantity boxes
            """)

if __name__ == "__main__":
    main()
