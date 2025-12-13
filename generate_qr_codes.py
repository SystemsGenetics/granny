#!/usr/bin/env python3
"""
Interactive QR Code Generator for Granny Tray Labels

Generates QR codes containing:
- Project Code
- Lot Code
- Date
- Variety

Format: PROJECT|LOT|DATE|VARIETY
Example: APPLE2025|LOT001|2025-12-02|BB-Late
"""

import qrcode
import os
from datetime import datetime


def generate_qr_code(project, lot, date, variety, output_dir="qr_codes"):
    """
    Generate a QR code with experimental information.

    Args:
        project: Project code (e.g., "APPLE2025")
        lot: Lot code (e.g., "LOT001")
        date: Date string (e.g., "2025-12-02")
        variety: Variety code (e.g., "BB-Late")
        output_dir: Directory to save QR codes
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create pipe-delimited data string
    qr_data = f"{project}|{lot}|{date}|{variety}"

    # Generate QR code
    qr = qrcode.QRCode(
        version=1,  # Controls size (1-40, 1 is smallest)
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,  # Size of each box in pixels
        border=4,  # Border size in boxes
    )

    qr.add_data(qr_data)
    qr.make(fit=True)

    # Create image
    img = qr.make_image(fill_color="black", back_color="white")

    # Create filename: PROJECT_LOT_DATE.png
    filename = f"{project}_{lot}_{date}.png"
    filepath = os.path.join(output_dir, filename)

    # Save image
    img.save(filepath)

    return filepath, qr_data


def main():
    """Interactive QR code generation"""
    print("=" * 60)
    print("QR Code Generator for Granny Tray Labels")
    print("=" * 60)
    print()

    while True:
        print("\nEnter information for QR code:")
        print("-" * 40)

        # Get user input
        project = input("Project Code (e.g., APPLE2025): ").strip()
        lot = input("Lot Code (e.g., LOT001): ").strip()

        # Date with default option
        date_input = input(f"Date (YYYY-MM-DD) [today: {datetime.now().strftime('%Y-%m-%d')}]: ").strip()
        date = date_input if date_input else datetime.now().strftime('%Y-%m-%d')

        variety = input("Variety (e.g., BB-Late): ").strip()

        # Validate inputs
        if not all([project, lot, date, variety]):
            print("\n❌ Error: All fields are required!")
            continue

        # Generate QR code
        try:
            filepath, qr_data = generate_qr_code(project, lot, date, variety)

            print("\n✅ QR Code Generated Successfully!")
            print(f"   Data: {qr_data}")
            print(f"   Saved to: {filepath}")

        except Exception as e:
            print(f"\n❌ Error generating QR code: {e}")
            continue

        # Ask if user wants to generate another
        print()
        again = input("Generate another QR code? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            break

    print("\n" + "=" * 60)
    print("Done! Check the 'qr_codes' folder for your QR codes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
