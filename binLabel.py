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

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Detect bus model and return quantity for the appropriate model
    Returns a dictionary with keys '7M', '9M', '12M' and their respective quantities
    """
    # Initialize result dictionary
    result = {'7M': '', '9M': '', '12M': ''}
    
    # Get quantity value
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = str(row[qty_veh_col])
    
    if not qty_veh:
        return result
    
    # Method 1: Check if there's a dedicated bus model column
    bus_model = ""
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        bus_model = str(row[bus_model_col]).upper()
    
    # Method 2: Check if bus model is embedded in the qty_veh value itself
    # Look for patterns like "7M:5", "9M-3", "12M 2", etc.
    qty_pattern = r'(\d+M)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    
    if matches:
        # If we found model-quantity pairs in the qty_veh field
        for model, quantity in matches:
            if model in result:
                result[model] = quantity
    elif bus_model:
        # If we have a separate bus model column, use that
        if '7M' in bus_model or '7' in bus_model:
            result['7M'] = qty_veh
        elif '9M' in bus_model or '9' in bus_model:
            result['9M'] = qty_veh
        elif '12M' in bus_model or '12' in bus_model:
            result['12M'] = qty_veh
    else:
        # Method 3: Try to infer from other columns or default behavior
        # Check if any other column contains bus model information
        for col_name, value in row.items():
            if pd.notna(value):
                value_str = str(value).upper()
                if '7M' in value_str:
                    result['7M'] = qty_veh
                    break
                elif '9M' in value_str:
                    result['9M'] = qty_veh
                    break
                elif '12M' in value_str:
                    result['12M'] = qty_veh
                    break
        
        # If no model detected, check for numeric patterns that might indicate model
        if not any(result.values()):
            # Look for common patterns in part numbers or descriptions that might indicate bus model
            part_desc = ""
            for col in row.index:
                if any(keyword in str(col).upper() for keyword in ['PART', 'DESC', 'NAME']):
                    if pd.notna(row[col]):
                        part_desc += str(row[col]).upper() + " "
            
            if '7M' in part_desc or '7 M' in part_desc:
                result['7M'] = qty_veh
            elif '9M' in part_desc or '9 M' in part_desc:
                result['9M'] = qty_veh
            elif '12M' in part_desc or '12 M' in part_desc:
                result['12M'] = qty_veh
            else:
                # Default: if no model detected, assume it's for the most common model (9M)
                result['9M'] = qty_veh
    
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
        canvas.setLineWidth(1.5)  # Make border slightly thicker
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
                    df = pd.read_csv(excel_file_path, encoding='latin1')

        if status_placeholder:
            status_placeholder.success(f"Successfully read file with {len(df)} rows")
            st.info(f"Columns found: {df.columns.tolist()}")
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if status_placeholder:
            status_placeholder.error(error_msg)
        return None

    # Identify columns (case-insensitive)
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    # Find relevant columns
    part_no_col = next((col for col in cols if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                   next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))

    desc_col = next((col for col in cols if 'DESC' in col),
                   next((col for col in cols if 'NAME' in col), cols[1] if len(cols) > 1 else part_no_col))

    # Look specifically for "QTY/BIN" column first, then fall back to general QTY column
    qty_bin_col = next((col for col in cols if 'QTY/BIN' in col or 'QTY_BIN' in col or 'QTYBIN' in col), 
                  next((col for col in cols if 'QTY' in col and 'BIN' in col), None))
    
    # If no specific QTY/BIN column is found, fall back to general QTY column
    if not qty_bin_col:
        qty_bin_col = next((col for col in cols if 'QTY' in col and 'VEH' not in col and 'BUS' not in col),
                      next((col for col in cols if 'QUANTITY' in col), None))
  
    loc_col = next((col for col in cols if 'LOC' in col or 'POS' in col or 'LOCATION' in col),
                   cols[2] if len(cols) > 2 else desc_col)

    # Improved detection of QTY/VEH column
    qty_veh_col = next((col for col in cols if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)

    # Look for bus model column
    bus_model_col = next((col for col in cols if any(term in col for term in ['BUS', 'MODEL', 'VEHICLE', 'TYPE'])), None)

    # Look for store location column
    store_loc_col = next((col for col in cols if 'STORE' in col and 'LOC' in col),
                      next((col for col in cols if 'STORELOCATION' in col), None))

    if status_placeholder:
        st.info(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            st.info(f"Qty/Veh Column: {qty_veh_col}")
        if bus_model_col:
            st.info(f"Bus Model Column: {bus_model_col}")
        if store_loc_col:
            st.info(f"Store Location Column: {store_loc_col}")

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
            progress_value = int((index+1)/total_rows*100)
            progress_bar.progress(progress_value)
        if status_placeholder:
            status_placeholder.info(f"Creating sticker {index+1} of {total_rows} ({int((index+1)/total_rows*100)}%)")
        
        elements = []

        # Extract data
        part_no = str(row[part_no_col])
        desc = str(row[desc_col])
        
        # Extract QTY/BIN properly
        qty_bin = ""
        if qty_bin_col and qty_bin_col in row and pd.notna(row[qty_bin_col]):
            qty_bin = str(row[qty_bin_col])
            
        # Detect bus model and get quantities for each model
        bus_model_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)
        
        location_str = str(row[loc_col]) if loc_col and loc_col in row else ""
        store_location = str(row[store_loc_col]) if store_loc_col and store_loc_col in row else ""
        location_parts = parse_location_string(location_str)

        # Generate QR code with part information
        qr_data = f"Part No: {part_no}\nDescription: {desc}\nLocation: {location_str}\n"
        qr_data += f"Store Location: {store_location}\n"
        for model, qty in bus_model_quantities.items():
            if qty:
                qr_data += f"{model}: {qty}\n"
        qr_data += f"QTY/BIN: {qty_bin}"
        
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
            ["Qty/Bin", Paragraph(str(qty_bin), qty_style)]
        ]

        # Create main table
        main_table = Table(main_table_data,
                         colWidths=[content_width/3, content_width*2/3],
                         rowHeights=[header_row_height, desc_row_height, qty_row_height])

        main_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
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
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        store_loc_table = Table(
            [[store_loc_label, store_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        store_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(store_loc_table)

        # Line Location section
        line_loc_label = Paragraph("Line Location", ParagraphStyle(
            name='LineLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))
        
        # The inner table width is already calculated above
        
        # Create the inner table
        line_loc_inner_table = Table(
            [location_parts],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )
        
        line_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 9)
        ]))
        
        # Wrap the label and the inner table in a containing table
        line_loc_table = Table(
            [[line_loc_label, line_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        line_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(line_loc_table)

        # Add smaller spacer between line location and bottom section
        elements.append(Spacer(1, 0.3*cm))

        # Bottom section - Bus model boxes with smart quantity placement
        mtm_box_width = 1.2*cm
        mtm_row_height = 1.5*cm

        # Use the detected bus model quantities
        position_matrix_data = [
            ["7M", "9M", "12M"],
            [bus_model_quantities['7M'], bus_model_quantities['9M'], bus_model_quantities['12M']]
        ]

        mtm_table = Table(
            position_matrix_data,
            colWidths=[mtm_box_width, mtm_box_width, mtm_box_width],
            rowHeights=[mtm_row_height/2, mtm_row_height/2]
        )

        mtm_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
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
        if status_placeholder:
            status_placeholder.success(f"PDF generated successfully: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        error_msg = f"Error building PDF: {e}"
        if status_placeholder:
            status_placeholder.error(error_msg)
            import traceback
            traceback.print_exc()
        return None

def main():
    # Check dependencies first
    if not check_dependencies():
        st.error("Failed to install required dependencies. Please check your environment.")
        return
    
    # Re-import after ensuring packages are available
    global PIL_AVAILABLE, QR_AVAILABLE
    try:
        from PIL import Image as PILImage
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        
    try:
        import qrcode
        QR_AVAILABLE = True
    except ImportError:
        QR_AVAILABLE = False
    
    st.title("🏷️ Bin Label Generator")
    
    # Add credit line below title
    st.markdown(
    """
    <div style='position: fixed; bottom: 20px; left: 20px; color: #666; font-size: 18px; font-weight: bold; z-index: 1000;'>
        <em>Designed and Developed by Agilomatrix</em>
    </div>
    """, 
    unsafe_allow_html=True
    )
    
    st.markdown("Generate professional sticker labels with QR codes from your Excel/CSV data")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Input Files")
        
        uploaded_file = st.file_uploader(
            "Upload Excel/CSV File",
            type=['xlsx', 'xls', 'csv'],
            help="Select your Excel or CSV file containing the data for label generation"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Show file info
            st.info(f"📊 File size: {uploaded_file.size} bytes")
            
    # Main content area
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📋 File Preview")
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            try:
                # Read and display preview
                if uploaded_file.name.lower().endswith('.csv'):
                    df_preview = pd.read_csv(temp_file_path)
                else:
                    df_preview = pd.read_excel(temp_file_path)
                
                st.dataframe(df_preview.head(10), use_container_width=True)
                st.info(f"📈 Total rows: {len(df_preview)} | Columns: {len(df_preview.columns)}")
                
                # Show column information
                with st.expander("📊 Column Information"):
                    st.write("**Available Columns:**")
                    for i, col in enumerate(df_preview.columns):
                        st.write(f"{i+1}. {col}")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                st.stop()
        
        with col2:
            st.header("⚙️ Generation Settings")
            
            # Generate button
            if st.button("🚀 Generate Sticker Labels", type="primary", use_container_width=True):
                # Create placeholders for status updates
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                with st.spinner("Generating PDF with QR codes..."):
                    # Create temporary output file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_output:
                        output_path = tmp_output.name
                    
                    # Create progress bar
                    progress_bar = progress_placeholder.progress(0)
                    
                    # Generate the PDF
                    result = generate_sticker_labels(
                        temp_file_path, 
                        output_path, 
                        progress_bar=progress_bar,
                        status_placeholder=status_placeholder
                    )
                    
                    if result:
                        # Success - offer download
                        status_placeholder.success("✅ PDF generation completed successfully!")
                        progress_bar.progress(100)
                        
                        # Read the generated PDF
                        with open(output_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                        
                        # Provide download button
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=f"{uploaded_file.name.split('.')[0]}_stickers.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        # Show success message
                        st.success("🎉 Your sticker labels are ready for download!")
                        
                        # Clean up temporary files
                        try:
                            os.unlink(output_path)
                        except:
                            pass
                    else:
                        status_placeholder.error("❌ PDF generation failed. Please check your file and try again.")
                
                # Clean up temporary input file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            # Additional information
            st.markdown("---")
            st.markdown("### ℹ️ Information")
            st.markdown("""
            **Features:**
            - ✅ Automatic QR code generation
            - ✅ Smart bus model detection (7M, 9M, 12M)
            - ✅ Professional sticker layout with bordered content boxes
            - 📊 Support for multiple data formats (Excel, CSV)
            - 🔍 Smart column detection for part numbers, descriptions, locations, and quantities
            - 📱 Mobile-friendly interface
            
            **Get started by uploading your file in the sidebar!**
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🏷️ Bin Label Generator | Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
