# Vehicle Load Predictor — Google Apps Script

This folder is a standalone Apps Script web-app version of the Streamlit dashboard.
It reads the same spreadsheet:

`1SbLc5pt0YPDBEQVOaOfyd-AJfvhTthQ5zUAcGgFU7Tc`

## Setup

1. Open [script.google.com](https://script.google.com) and create a new Apps Script project.
2. Add two files:
   - `Code.gs` — copy `Vehicle_Load_Predictor_Apps_Script/Code.gs`
   - `Index.html` — copy `Vehicle_Load_Predictor_Apps_Script/Index.html`
3. Confirm the Apps Script account can access the spreadsheet.
4. Click **Deploy → New deployment**.
5. Select **Web app**.
6. Choose who should have access, then deploy and open the web-app URL.

The app uses only Bag, Semi-Large, and Tote sheets for load data. It takes DH names
from raw destinations, then gets cutoff/code from `DH Name Cut-Off Wise`.
Vehicle-size limits are applied only for exact normalized DH-name matches in the
Vehicle Capacity sheet; unmatched DHs remain unrestricted.

The web app includes cutoff filtering, multi-DH selection, vehicle prediction,
Ready to Dispatch, vehicle-capacity reference, refresh, and CSV export.
