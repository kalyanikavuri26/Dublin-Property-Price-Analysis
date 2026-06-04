#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# These first two lines are special instructions for the system:
# 1. #!/usr/bin/env python3 tells the computer: "Run this file using Python 3"
# 2. # -*- coding: utf-8 -*- tells Python: "This file uses UTF-8 text encoding"

###############################################################################
# Dublin Property Value Analyzer Dashboard (Dash + Plotly)
###############################################################################
# PURPOSE (simple English):
#   This program creates an interactive web dashboard to analyze Dublin property
#   sale prices (2020–2025). It lets you filter properties and see:
#     - Price distribution
#     - Price vs size and distance charts
#     - Distance impact to city center / parks / schools / transport
#     - A correlation heatmap (relationship between variables)
#     - A map of property points
#
# HOW TO RUN:
#   1) Install dependencies:
#        pip install dash dash-bootstrap-components pandas numpy plotly scipy
#        (optional) pip install flask-caching pyarrow
#   2) Ensure these files are in same folder:
#        - properties_with_park_distance.csv
#        - schools_colleges_universities_with_lat_lon_unique.csv   (optional)
#        - transportation_bus_and_luas_stops_dataset.csv           (optional)
#   3) Run:
#        python3 dublin_property_dashboard.py
#   4) Open in browser:
#        http://localhost:8050
###############################################################################

# -----------------------------------------------------------------------------
# SECTION 1: IMPORTING LIBRARIES (bringing in tools we need)
# -----------------------------------------------------------------------------
# We import libraries like getting tools from a toolbox

import hashlib             # Tool for creating unique codes (hashes) for caching
import warnings            # Tool for hiding warning messages in the console
warnings.filterwarnings("ignore")  # Tell warnings library: "Don't show warnings"

# Numerical and data processing libraries:
import numpy as np         # Math toolbox: arrays, calculations, statistics
import pandas as pd        # Data toolbox: read CSV, work with tables, clean data

# Dashboard and web interface libraries:
import dash                # Main dashboard framework (creates web app)
# Import specific parts from dash:
from dash import dcc       # dash_core_components: graphs, sliders, dropdowns
from dash import html      # HTML components: divs, headers, buttons
from dash import Input     # For user input (slider moves, dropdown clicks)
from dash import Output    # For updating charts when user interacts
from dash import State     # For reading current state of components
from dash import dash_table  # For displaying data tables
from dash.exceptions import PreventUpdate  # For stopping unnecessary updates

import dash_bootstrap_components as dbc  # Pre-made UI components (nice buttons, cards)

# Charting libraries:
import plotly.express as px    # Easy chart creation (simple commands for charts)
import plotly.graph_objects as go  # Advanced chart customization

# Specialized libraries:
from scipy.spatial import cKDTree  # Fast search for nearest points (for distances)

# -----------------------------------------------------------------------------
# SECTION 2: TRY TO IMPORT OPTIONAL CACHE (speed up repeated requests)
# -----------------------------------------------------------------------------
# Cache stores results so we don't have to rebuild charts for same filters
try:
    from flask_caching import Cache  # Try to import caching library
    CACHE_AVAILABLE = True           # If successful, mark cache as available
except Exception:                    # If import fails (library not installed)
    CACHE_AVAILABLE = False          # Mark cache as unavailable


###############################################################################
# SECTION 3: CONSTANTS (fixed values used throughout the program)
###############################################################################

# Dublin City Center coordinates (latitude/longitude):
# These exact numbers represent Dublin's center on the map
CITY_CENTER_LAT = 53.349805   # Latitude (north-south position)
CITY_CENTER_LON = -6.260310   # Longitude (east-west position)

# Earth's radius in kilometers:
# Used in distance calculations on Earth's curved surface
EARTH_RADIUS_KM = 6371.0088

# Column name constants (avoid typos by using variables):
PRICE_COL = "Price_clean"                    # Column name for property price
SIZE_COL = "Property Size(Square Meters)"    # Column name for property size
ADDRESS_COL = "Address"                      # Column name for property address
YEAR_COL = "sale_year"                       # Column name for sale year

# Year range for the dashboard:
YEAR_UI_MIN = 2020          # Minimum year to show
YEAR_UI_MAX = 2025          # Maximum year to show
YEAR_UI_DEFAULT = [2020, 2025]  # Default year range (both sliders start here)

# Dublin area boundaries (to filter out properties outside Dublin):
DUBLIN_LAT_MIN, DUBLIN_LAT_MAX = 53.20, 53.45   # South to North boundaries
DUBLIN_LON_MIN, DUBLIN_LON_MAX = -6.45, -6.05   # West to East boundaries


###############################################################################
# SECTION 4: "NO DATA" LABELS (for missing information)
###############################################################################
# When distance data is missing, we show these labels instead
NO_CITY = "No city data"          # Shown when city distance data is missing
NO_PARK = "No park data"          # Shown when park distance data is missing
NO_SCHOOL = "No school data"      # Shown when school distance data is missing
NO_TRANSPORT = "No transport data"  # Shown when transport distance data is missing


###############################################################################
# SECTION 5: CATEGORY ORDERS (correct display order for distance buckets)
###############################################################################
# These lists ensure distance categories appear in logical order on charts
# Example: "<2km" should come before "2-5km", not after it

# City center distance categories in correct order:
CITY_ORDER = ["<2km", "2-5km", "5-10km", "10-15km", "15-20km", ">20km", NO_CITY]

# Park distance categories in correct order:
PARK_ORDER = ["<0.5km", "0.5-1km", "1-2km", "2-5km", "5-10km", ">10km", NO_PARK]

# School distance categories in correct order:
SCHOOL_ORDER = ["<0.5km", "0.5-1km", "1-2km", "2-5km", "5-10km", ">10km", NO_SCHOOL]

# Transport distance categories in correct order:
TRANSPORT_ORDER = ["<0.2km", "0.2-0.5km", "0.5-1km", "1-2km", "2-5km", ">5km", NO_TRANSPORT]

# Dictionary to quickly find the right order list for any column:
CATEGORY_ORDERS = {
    "city_proximity": CITY_ORDER,        # Map column name to its order list
    "park_proximity": PARK_ORDER,        # Map column name to its order list
    "school_proximity": SCHOOL_ORDER,    # Map column name to its order list
    "transport_proximity": TRANSPORT_ORDER,  # Map column name to its order list
}


###############################################################################
# SECTION 6: HELPER FUNCTIONS (small tools that do specific jobs)
###############################################################################

def safe_numeric(df: pd.DataFrame, col: str) -> None:
    """
    Convert a dataframe column to numeric safely.
    
    How it works:
    1. Takes a dataframe (table) and column name
    2. Tries to convert all values in that column to numbers
    3. If a value can't be converted (like "NA" or "unknown"), it becomes NaN (missing)
    4. This prevents math errors when we try to calculate with text values
    
    Example:
      Input column: ["100000", "200000", "NA", "unknown", "300000"]
      Output column: [100000.0, 200000.0, NaN, NaN, 300000.0]
    """
    if col in df.columns:  # Only if column exists in the dataframe
        # pd.to_numeric tries to convert, errors="coerce" changes bad values to NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")


def haversine_km(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two points on Earth using Haversine formula.
    
    Why Haversine? Because Earth is round, not flat!
    Straight-line distance on a map doesn't account for Earth's curvature.
    
    Steps:
    1. Convert degrees to radians (math functions need radians, not degrees)
    2. Calculate differences in longitude and latitude
    3. Apply Haversine formula
    4. Convert result to kilometers
    
    Input: Longitude1, Latitude1, Longitude2, Latitude2
    Output: Distance in kilometers
    """
    # Step 1: Convert all angles from degrees to radians
    # map() applies np.radians to each value in the list
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Step 2: Calculate differences
    dlon = lon2 - lon1  # Difference in longitude
    dlat = lat2 - lat1  # Difference in latitude
    
    # Step 3: Haversine formula
    # a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    
    # Step 4: Angular distance in radians
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Step 5: Convert to kilometers
    return EARTH_RADIUS_KM * c


def latlon_to_unit_xyz(lat_deg, lon_deg):
    """
    Convert (latitude, longitude) to 3D coordinates on a unit sphere.
    
    Why convert to 3D?
    - KDTree (our fast search tool) works best in regular 3D space
    - On Earth's surface, we need to account for curvature
    - By converting to 3D, we can use straight-line distance in 3D space
    
    Math behind it:
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    
    Returns: Array of [x, y, z] coordinates
    """
    # Convert degrees to radians
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    
    # Calculate 3D coordinates
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    
    # Stack into Nx3 array: [[x1,y1,z1], [x2,y2,z2], ...]
    return np.column_stack([x, y, z])


def nearest_distance_km_sphere(points_latlon: np.ndarray, ref_latlon: np.ndarray) -> np.ndarray:
    """
    Find the nearest distance from each property to reference points (schools, transport).
    
    Steps:
    1. Convert all points to 3D coordinates
    2. Build a KDTree from reference points (fast search structure)
    3. For each property, find closest reference point
    4. Convert 3D distance back to Earth surface distance
    
    Inputs:
      points_latlon: Property locations [[lat1, lon1], [lat2, lon2], ...]
      ref_latlon: Reference points [[lat1, lon1], [lat2, lon2], ...]
    
    Returns: Array of distances in km for each property
    """
    # Step 1: Check if we have reference points
    if ref_latlon is None or len(ref_latlon) == 0:
        # No reference points: return all NaN (missing values)
        return np.full((len(points_latlon),), np.nan)
    
    # Step 2: Convert reference points to 3D and build KDTree
    ref_xyz = latlon_to_unit_xyz(ref_latlon[:, 0], ref_latlon[:, 1])
    tree = cKDTree(ref_xyz)  # Build fast search structure
    
    # Step 3: Convert property points to 3D and query nearest neighbors
    pts_xyz = latlon_to_unit_xyz(points_latlon[:, 0], points_latlon[:, 1])
    # Query finds nearest distance in 3D space (chord distance)
    chord_dist, _ = tree.query(pts_xyz, k=1)
    
    # Step 4: Convert 3D chord distance to arc distance on sphere
    # Clip ensures value is between 0 and 2 (mathematical requirement)
    chord_dist = np.clip(chord_dist, 0, 2)
    # Calculate angle: θ = 2 * arcsin(chord_dist / 2)
    theta = 2 * np.arcsin(chord_dist / 2.0)
    
    # Step 5: Convert to kilometers
    return EARTH_RADIUS_KM * theta


def compute_distance_category(dist_km, feature_name):
    """
    Convert numeric distance to a category label (bucket).
    
    Example:
      1.2 km to city center -> "<2km"
      7.5 km to park -> "5-10km"
      NaN (missing) -> "No city data"
    
    Different features have different bucket ranges:
      - Transport: very close ranges (0.2km, 0.5km)
      - Schools/Parks: medium ranges (0.5km, 1km, 2km)
      - City: larger ranges (2km, 5km, 10km)
    """
    # Step 1: Check for missing distance (NaN)
    if pd.isna(dist_km):
        # Return appropriate "no data" label based on feature
        if feature_name == "city_center":
            return NO_CITY
        if feature_name == "park":
            return NO_PARK
        if feature_name == "school":
            return NO_SCHOOL
        if feature_name == "transport":
            return NO_TRANSPORT
        return "No data"  # Fallback
    
    # Step 2: Transport distance buckets (very close ranges)
    if feature_name == "transport":
        if dist_km <= 0.2:
            return "<0.2km"
        elif dist_km <= 0.5:
            return "0.2-0.5km"
        elif dist_km <= 1:
            return "0.5-1km"
        elif dist_km <= 2:
            return "1-2km"
        elif dist_km <= 5:
            return "2-5km"
        else:
            return ">5km"
    
    # Step 3: School and Park distance buckets (medium ranges)
    if feature_name in ("school", "park"):
        if dist_km <= 0.5:
            return "<0.5km"
        elif dist_km <= 1:
            return "0.5-1km"
        elif dist_km <= 2:
            return "1-2km"
        elif dist_km <= 5:
            return "2-5km"
        elif dist_km <= 10:
            return "5-10km"
        else:
            return ">10km"
    
    # Step 4: City center distance buckets (larger ranges)
    if dist_km <= 2:
        return "<2km"
    elif dist_km <= 5:
        return "2-5km"
    elif dist_km <= 10:
        return "5-10km"
    elif dist_km <= 15:
        return "10-15km"
    elif dist_km <= 20:
        return "15-20km"
    else:
        return ">20km"


def normalize_category_series(s: pd.Series) -> pd.Series:
    """
    Clean up category strings by removing extra spaces.
    
    Why needed?
    - Data might have: " >20km" or ">20km " (with spaces)
    - These are technically different from ">20km" (no spaces)
    - Cleaning ensures proper grouping and ordering
    
    Example:
      Input: [" <2km", "2-5km ", ">20km"]
      Output: ["<2km", "2-5km", ">20km"]
    """
    # .astype(str): Convert to string (in case of numbers)
    # .str.strip(): Remove spaces from beginning and end
    return s.astype(str).str.strip()


def present_categories(df_in: pd.DataFrame, col: str, full_order: list) -> list:
    """
    Get only the categories that actually exist in current filtered data.
    
    Why needed?
    - Plotly crashes if we ask for categories that don't exist
    - Example: Full order has 7 categories, but filtered data only has 3
    - We return only those 3 categories, in the correct order
    
    Returns: List of categories present in data, in correct order
    """
    # Step 1: Check if column exists
    if col not in df_in.columns:
        return []  # Column doesn't exist
    
    # Step 2: Get unique categories in data (remove missing values, strip spaces)
    present = set(df_in[col].dropna().astype(str).str.strip().unique())
    
    # Step 3: Return only categories that are both in full_order AND present
    return [c for c in full_order if c in present]


def apply_x_category_order(fig, order_list):
    """
    Force Plotly x-axis to use specific order for categories.
    
    Alternative to category_orders parameter in px.bar()
    This method is safer and avoids Plotly internal errors.
    """
    if order_list:  # Only if we have an order list
        # categoryorder="array": Use custom order
        # categoryarray=order_list: Use this specific order
        fig.update_xaxes(categoryorder="array", categoryarray=order_list)
    return fig


def order_dataframe_by_category(df_in: pd.DataFrame, col: str, full_order: list) -> pd.DataFrame:
    """
    Sort dataframe rows based on category order.
    
    Steps:
    1. Create mapping: category -> position in order
    2. Add temporary column with order index
    3. Sort by order index
    4. Remove temporary column
    
    Example:
      Input: df with categories in random order ["5-10km", "<2km", "2-5km"]
      Output: df sorted as ["<2km", "2-5km", "5-10km"]
    """
    # Step 1: Check inputs
    if col not in df_in.columns or not full_order:
        return df_in  # Can't sort
    
    # Step 2: Create mapping dictionary
    # {"<2km": 0, "2-5km": 1, "5-10km": 2, ...}
    idx_map = {v: i for i, v in enumerate(full_order)}
    
    # Step 3: Create copy to avoid modifying original
    tmp = df_in.copy()
    
    # Step 4: Clean category column
    tmp[col] = normalize_category_series(tmp[col])
    
    # Step 5: Create order index column
    # .map(idx_map): Convert category to its position (0, 1, 2, ...)
    # .fillna(9999): If category not in order list, use large number (puts at end)
    # .astype(int): Ensure it's integer
    tmp["_order_idx"] = tmp[col].map(idx_map).fillna(9999).astype(int)
    
    # Step 6: Sort and remove helper column
    tmp = tmp.sort_values("_order_idx").drop(columns=["_order_idx"])
    return tmp


def empty_fig(msg="No data for selected filters"):
    """
    Create a blank figure with a message.
    
    Used when:
      - No data matches filters
      - Data is missing
      - Error occurs
    
    Returns: Figure with centered text message
    """
    fig = go.Figure()  # Create empty figure
    # Add annotation (text) in center
    fig.add_annotation(
        text=msg,           # Message text
        x=0.5, y=0.5,      # Center position (0.5, 0.5 is middle)
        showarrow=False,    # No arrow pointing to text
        font=dict(size=16)  # Text size
    )
    # Set layout
    fig.update_layout(
        template="plotly_white",  # White background theme
        height=350,               # Figure height in pixels
        margin=dict(l=10, r=10, t=40, b=10)  # Margins
    )
    return fig


def fig_layout(fig, height=None):
    """
    Apply consistent layout to all figures.
    
    Features:
      - White background
      - Proper title positioning
      - Legend below title
      - Consistent margins
    
    Why needed?
      - Makes all charts look consistent
      - Prevents title/legend overlap
    """
    # Basic layout settings
    fig.update_layout(
        template="plotly_white",  # Clean white theme
        margin=dict(l=0, r=10, t=20, b=10),  # Margins (top=20 for title space)
        hovermode="closest",  # Show info for closest point on hover
        
        # Title settings
        title=dict(
            x=0.5,              # Center horizontally
            xanchor="center",   # Anchor at center
            y=0.98,             # Near top (0.98 = 98% from bottom)
            yanchor="top",      # Anchor at top of title
            font=dict(size=18), # Title font size
        ),
        
        # Legend settings
        legend=dict(
            orientation="h",    # Horizontal legend
            yanchor="top",      # Anchor at top
            y=0.99,             # Position (just below title)
            xanchor="center",   # Center horizontally
            x=0.5,              # Center position
            title_text=""       # No legend title
        ),
    )
    
    # Set height if provided
    if height is not None:
        fig.update_layout(height=height)
    
    return fig


def sample_df(df_in: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Randomly sample rows from dataframe if it's too large.
    
    Why sample?
      - Scatter plots with 100,000 points are slow
      - 15,000 points look almost the same but render faster
      - Random sampling preserves overall patterns
    
    Returns: Sampled dataframe with n rows (or all if df is smaller)
    """
    if len(df_in) <= n:  # If dataframe already small enough
        return df_in
    # Return random sample of n rows
    # random_state=seed ensures same sample each time (for consistency)
    return df_in.sample(n=n, random_state=seed)


def metric_label(metric: str) -> str:
    """
    Create user-friendly label for selected metric.
    
    Example:
      Input: "Price_clean"
      Output: "Price (€)"
      
      Input: "price_per_sqm"
      Output: "Price per m² (€)"
    """
    # If metric is price column, return "Price (€)"
    # Otherwise return "Price per m² (€)"
    return "Price (€)" if metric == PRICE_COL else "Price per m² (€)"


def build_correlation_heatmap(df_in: pd.DataFrame):
    """
    Build correlation matrix heatmap.
    
    What is correlation?
      - Measures relationship between two variables
      - Range: -1 to 1
      - 1.0: Perfect positive relationship (both increase together)
      - -1.0: Perfect negative relationship (one increases, other decreases)
      - 0.0: No relationship
    
    Steps:
    1. Select numeric columns
    2. Calculate correlation matrix
    3. Create heatmap with colors
    4. Add correlation values as text
    
    Returns: Figure showing correlation heatmap
    """
    # Step 1: List of columns we want in correlation matrix
    wanted = [
        PRICE_COL,              # Property price
        "price_per_sqm",        # Price per square meter
        SIZE_COL,               # Property size
        "dist_to_city_center_km",  # Distance to city center
        "dist_to_park_km",      # Distance to nearest park
        "dist_to_school_km",    # Distance to nearest school
        "dist_to_transport_km", # Distance to nearest transport
    ]
    
    # Step 2: Keep only columns that exist in dataframe
    cols = [c for c in wanted if c in df_in.columns]
    
    # Step 3: Check we have enough columns
    if len(cols) < 2:
        return empty_fig("Not enough numeric columns for correlation heatmap")
    
    # Step 4: Create copy and convert to numeric
    d = df_in[cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    
    # Step 5: Remove infinite values and missing values
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Step 6: Check we have enough rows
    if len(d) < 10:
        return empty_fig("Not enough rows after filtering for correlation heatmap")
    
    # Step 7: Calculate correlation matrix
    corr = d.corr(method="pearson")
    
    # Step 8: Get column names and values for heatmap
    x = corr.columns.tolist()  # Column names for x-axis
    y = corr.index.tolist()    # Row names for y-axis
    z = corr.values            # Correlation values
    
    # Step 9: Create heatmap figure
    fig = go.Figure(
        data=go.Heatmap(
            z=z,                    # Correlation values
            x=x,                    # X-axis labels
            y=y,                    # Y-axis labels
            zmin=-1, zmax=1,       # Color scale range (-1 to 1)
            colorscale="RdBu",     # Red-Blue color scale
            reversescale=True,     # Blue for positive, Red for negative
            colorbar=dict(title="Corr"),  # Color bar label
            # Custom hover text
            hovertemplate="X=%{x}<br>Y=%{y}<br>Corr=%{z:.2f}<extra></extra>",
        )
    )
    
    # Step 10: Add correlation values as text on heatmap
    xs, ys, ts = [], [], []  # Lists to store positions and text
    
    # Loop through all cells
    for i, yy in enumerate(y):      # For each row
        for j, xx in enumerate(x):  # For each column
            xs.append(xx)           # X position
            ys.append(yy)           # Y position
            ts.append(f"{z[i, j]:.2f}")  # Correlation value (2 decimal places)
    
    # Add text trace
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,                # Positions
            mode="text",               # Display as text (not markers/lines)
            text=ts,                   # Text values
            textfont=dict(size=12, color="black"),  # Font settings
            hoverinfo="skip",          # Don't show hover for text
            showlegend=False,          # Don't show in legend
        )
    )
    
    # Step 11: Set layout
    fig.update_layout(
        title="Co-relationship Heatmap (Correlation Matrix)",
        height=520,                    # Figure height
        margin=dict(l=40, r=20, t=60, b=40),  # Margins
        template="plotly_white",       # White theme
    )
    return fig


###############################################################################
# SECTION 7: LOAD & PROCESS DATA (most important part)
###############################################################################
def load_and_process_data() -> pd.DataFrame:
    """
    MAIN DATA PROCESSING FUNCTION
    
    Steps:
    1. Read properties data from CSV
    2. Clean and convert columns
    3. Filter to Dublin area (2020-2025)
    4. Compute distances to city center, parks, schools, transport
    5. Create distance categories (buckets)
    6. Return clean dataframe
    
    Returns: Processed pandas DataFrame ready for dashboard
    """
    print("Step 1: Reading properties data from CSV...")
    # Step 1: Read the main properties dataset
    props = pd.read_csv("properties_with_park_distance.csv")
    
    # Clean column names (remove extra spaces)
    props.columns = props.columns.str.strip()
    
    # Optional: Convert to Parquet format (faster for future loads)
    try:
        props.to_parquet("properties_with_park_distance.parquet", index=False, engine="pyarrow")
        print("✅ CSV converted to Parquet successfully!")
    except Exception:
        print("ℹ️ Parquet conversion skipped (pyarrow not available).")
    
    print("Step 2: Converting columns to numeric...")
    # Step 2: Convert important columns to numeric
    columns_to_convert = ["Price (€)", PRICE_COL, "lat", "lon", "dist_to_park_m", SIZE_COL, "Date of Sale (yyyy)"]
    for c in columns_to_convert:
        safe_numeric(props, c)  # Convert safely, bad values become NaN
    
    # Step 3: Ensure we have price column
    # If dataset uses "Price (€)" instead of "Price_clean", copy it
    if PRICE_COL not in props.columns and "Price (€)" in props.columns:
        props[PRICE_COL] = props["Price (€)"]
    
    # Step 4: Ensure we have address column
    if ADDRESS_COL not in props.columns:
        props[ADDRESS_COL] = "Unknown Address"  # Fill with placeholder
    
    print("Step 3: Removing properties without location data...")
    # Step 5: Remove rows without latitude/longitude (can't plot or compute distances)
    props = props.dropna(subset=["lat", "lon"]).copy()
    
    print("Step 4: Filtering to Dublin area...")
    # Step 6: Filter to Dublin bounding box
    props = props[
        (props["lat"].between(DUBLIN_LAT_MIN, DUBLIN_LAT_MAX)) &
        (props["lon"].between(DUBLIN_LON_MIN, DUBLIN_LON_MAX))
    ].copy()
    
    print("Step 5: Extracting sale year...")
    # Step 7: Extract sale year
    if "Date of Sale (yyyy)" in props.columns:
        # Convert to numeric, then to integer type (Int64 supports NaN)
        props[YEAR_COL] = pd.to_numeric(props["Date of Sale (yyyy)"], errors="coerce").astype("Int64")
    else:
        # If no date column, create empty year column
        props[YEAR_COL] = pd.Series([pd.NA] * len(props), dtype="Int64")
    
    # Step 8: Filter to years 2020-2025 only
    props = props[props[YEAR_COL].between(YEAR_UI_MIN, YEAR_UI_MAX)].copy()
    
    print("Step 6: Calculating distance to city center...")
    # Step 9: Calculate distance to Dublin city center using Haversine
    props["dist_to_city_center_km"] = haversine_km(
        props["lon"].values,   # Property longitudes
        props["lat"].values,   # Property latitudes
        CITY_CENTER_LON,       # City center longitude
        CITY_CENTER_LAT        # City center latitude
    )
    
    print("Step 7: Processing park distances...")
    # Step 10: Convert park distance from meters to kilometers
    props["dist_to_park_km"] = np.nan  # Initialize as missing
    if "dist_to_park_m" in props.columns:  # If park distance column exists
        props["dist_to_park_km"] = props["dist_to_park_m"] / 1000.0  # m to km
    
    print("Step 8: Calculating price per square meter...")
    # Step 11: Calculate price per square meter
    props["price_per_sqm"] = np.nan  # Initialize as missing
    if SIZE_COL in props.columns:
        # Replace 0 size with NaN (avoid division by zero)
        size = props[SIZE_COL].replace(0, np.nan)
        # Price / size = price per square meter
        props["price_per_sqm"] = props[PRICE_COL] / size
    
    print("Step 9: Preparing for school/transport distance calculations...")
    # Step 12: Extract property coordinates for distance calculations
    property_points = props[["lat", "lon"]].to_numpy()
    
    print("Step 10: Calculating distance to nearest school...")
    # Step 13: Calculate distance to nearest school
    try:
        # Try to load school data
        schools = pd.read_csv("schools_colleges_universities_with_lat_lon_unique.csv")
        schools.columns = schools.columns.str.strip()
        
        # Standardize column names
        if "Latitude" in schools.columns and "Longitude" in schools.columns:
            schools = schools.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        
        # Convert to numeric
        safe_numeric(schools, "lat")
        safe_numeric(schools, "lon")
        
        # Remove schools without location
        schools = schools.dropna(subset=["lat", "lon"])
        
        # Convert to numpy array
        school_points = schools[["lat", "lon"]].to_numpy()
        
        # Calculate nearest school distance for each property
        props["dist_to_school_km"] = nearest_distance_km_sphere(property_points, school_points)
        print(f"✅ School distances calculated for {len(school_points)} schools")
    except Exception as e:
        # If school file missing or error, keep NaN
        props["dist_to_school_km"] = np.nan
        print(f"⚠️ School data not available: {e}")
    
    print("Step 11: Calculating distance to nearest transport...")
    # Step 14: Calculate distance to nearest transport stop
    try:
        # Try to load transport data
        transport = pd.read_csv("transportation_bus_and_luas_stops_dataset.csv")
        transport.columns = transport.columns.str.strip()
        
        # Standardize column names
        if "latitude" in transport.columns and "longitude" in transport.columns:
            transport = transport.rename(columns={"latitude": "lat", "longitude": "lon"})
        if "Latitude" in transport.columns and "Longitude" in transport.columns:
            transport = transport.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        
        # Convert to numeric
        safe_numeric(transport, "lat")
        safe_numeric(transport, "lon")
        
        # Remove stops without location
        transport = transport.dropna(subset=["lat", "lon"])
        
        # Convert to numpy array
        transport_points = transport[["lat", "lon"]].to_numpy()
        
        # Calculate nearest transport distance for each property
        props["dist_to_transport_km"] = nearest_distance_km_sphere(property_points, transport_points)
        print(f"✅ Transport distances calculated for {len(transport_points)} stops")
    except Exception as e:
        # If transport file missing or error, keep NaN
        props["dist_to_transport_km"] = np.nan
        print(f"⚠️ Transport data not available: {e}")
    
    print("Step 12: Creating distance categories...")
    # Step 15: Convert numeric distances to category labels
    # City center categories
    props["city_proximity"] = props["dist_to_city_center_km"].apply(
        lambda x: compute_distance_category(x, "city_center")
    )
    
    # Park categories
    props["park_proximity"] = props["dist_to_park_km"].apply(
        lambda x: compute_distance_category(x, "park")
    )
    
    # School categories
    props["school_proximity"] = props["dist_to_school_km"].apply(
        lambda x: compute_distance_category(x, "school")
    )
    
    # Transport categories
    props["transport_proximity"] = props["dist_to_transport_km"].apply(
        lambda x: compute_distance_category(x, "transport")
    )
    
    print("Step 13: Cleaning category strings...")
    # Step 16: Clean category strings (remove extra spaces)
    for c in ["city_proximity", "park_proximity", "school_proximity", "transport_proximity"]:
        props[c] = normalize_category_series(props[c])
    
    print("Step 14: Final cleaning...")
    # Step 17: Final cleaning
    # Remove properties without price
    props = props.dropna(subset=[PRICE_COL]).copy()
    
    # Remove properties with price <= 0 (invalid)
    props = props[props[PRICE_COL] > 0].copy()
    
    # Reset index (0, 1, 2, ...)
    props = props.reset_index(drop=True)
    
    print(f"✅ Data loaded successfully: {len(props)} properties")
    return props


###############################################################################
# SECTION 8: LOAD DATA ONCE (global dataframe used throughout)
###############################################################################
print("\n" + "="*60)
print("LOADING DATA...")
print("="*60)

# Load and process data (this runs once when program starts)
df = load_and_process_data()

# Check if data is empty
if df.empty:
    print("❌ ERROR: No data found for years 2020–2025. Check dataset.")
    raise SystemExit("❌ No data found for years 2020–2025. Check dataset.")

print(f"\n✅ Data Summary:")
print(f"   - Total properties: {len(df):,}")
print(f"   - Year range: {int(df[YEAR_COL].min())} - {int(df[YEAR_COL].max())}")
print(f"   - Price range: €{df[PRICE_COL].min():,.0f} - €{df[PRICE_COL].max():,.0f}")
print("="*60)


###############################################################################
# SECTION 9: CALCULATE SLIDER RANGES
###############################################################################
# Calculate min/max values for sliders based on loaded data
PRICE_MIN = int(np.nanmin(df[PRICE_COL]))  # Minimum price in data
PRICE_MAX = int(np.nanmax(df[PRICE_COL]))  # Maximum price in data

# Calculate size range (if size column exists)
if SIZE_COL in df.columns and not df[SIZE_COL].dropna().empty:
    SIZE_MIN = int(np.nanmin(df[SIZE_COL]))  # Minimum size
    SIZE_MAX = int(np.nanmax(df[SIZE_COL]))  # Maximum size
else:
    SIZE_MIN, SIZE_MAX = 0, 300  # Default range


###############################################################################
# SECTION 10: SETUP DASH APP & CACHE
###############################################################################
print("\nSetting up dashboard...")

# Create Dash application instance
app = dash.Dash(
    __name__,  # Name of the app
    external_stylesheets=[dbc.themes.FLATLY],  # Bootstrap theme for nice UI
    suppress_callback_exceptions=True,  # Allow dynamic layouts without errors
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],  # Mobile friendly
)
app.title = "Dublin Property Value Analyzer"  # Browser tab title

# Setup cache (if available)
cache = None
if CACHE_AVAILABLE:
    # SimpleCache stores results in memory for 300 seconds (5 minutes)
    cache = Cache(app.server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})
    print("✅ Cache enabled (reuses charts for same filters)")
else:
    print("ℹ️ Cache not available (install flask_caching for better performance)")


###############################################################################
# SECTION 11: CACHE HELPER FUNCTIONS
###############################################################################
def _hash_obj(obj) -> str:
    """
    Create unique hash string from any Python object.
    
    Used for cache keys: same filters = same hash = same cache key
    """
    # Convert object to string, encode to bytes, create MD5 hash
    return hashlib.md5(str(obj).encode("utf-8")).hexdigest()


def cached_or_build(prefix: str, tab: str, metric: str, idx_list, build_fn):
    """
    Smart caching: reuse charts if same filters, otherwise build new.
    
    Steps:
    1. Create unique cache key from filters
    2. Check if chart exists in cache
    3. If yes, return cached chart
    4. If no, build chart, store in cache, return it
    """
    # If cache not available, just build the chart
    if not CACHE_AVAILABLE or cache is None:
        return build_fn()
    
    # Create cache key: combination of prefix, tab, metric, and filter hash
    key = f"{prefix}:{tab}:{metric}:{_hash_obj(idx_list)}"
    
    # Try to get from cache
    got = cache.get(key)
    if got is not None:
        return got  # Cache hit: return cached chart
    
    # Cache miss: build chart
    built = build_fn()
    
    # Store in cache for future use
    cache.set(key, built)
    
    return built


def df_from_idx(idx_list):
    """
    Convert list of row indexes to dataframe.
    
    Why store indexes instead of full dataframe?
      - Indexes are small (integers)
      - Full dataframe is large (slow to pass between callbacks)
      - We store indexes in dcc.Store, reconstruct dataframe when needed
    """
    if not idx_list:  # If empty list
        return df.iloc[0:0].copy()  # Return empty dataframe
    
    # Use .iloc to select rows by index
    return df.iloc[idx_list].copy()


###############################################################################
# SECTION 12: BUILD UI COMPONENTS (HTML/Dash components)
###############################################################################
print("Building UI components...")

# ============================================================================
# SIDEBAR (left panel with filters)
# ============================================================================
sidebar = html.Div(
    [
        # Sidebar header
        html.H4("🔍 Filters", className="mb-3"),
        html.Hr(),
        
        # Metric dropdown: Price or Price per m²
        html.H6("Metric"),
        dcc.Dropdown(
            id="metric",  # Unique ID for this component
            options=[  # Dropdown options
                {"label": "Price (€)", "value": PRICE_COL},
                {"label": "Price per m² (€)", "value": "price_per_sqm"},
            ],
            value=PRICE_COL,  # Default selection
            clearable=False,  # Can't clear selection
            className="mb-3",  # CSS class for spacing
        ),
        
        # Year range slider
        html.H6("Sale Year Range (2020–2025)"),
        dcc.RangeSlider(
            id="year-slider",
            min=YEAR_UI_MIN,
            max=YEAR_UI_MAX,
            value=YEAR_UI_DEFAULT,
            marks={y: str(y) for y in range(YEAR_UI_MIN, YEAR_UI_MAX + 1)},  # Tick marks
            step=1,  # Increment by 1 year
            updatemode="mouseup",  # Update only when mouse released
            className="mb-4",
        ),
        
        # Price range slider
        html.H6("Price Range (€)"),
        dcc.RangeSlider(
            id="price-slider",
            min=PRICE_MIN,
            max=PRICE_MAX,
            value=[PRICE_MIN, PRICE_MAX],  # Start with full range
            step=max(10000, int((PRICE_MAX - PRICE_MIN) / 300) or 10000),  # Smart step size
            marks=None,  # No fixed marks
            tooltip={"placement": "bottom", "always_visible": True},  # Show current values
            updatemode="mouseup",
            className="mb-4",
        ),
        
        # Size range slider
        html.H6("Property Size (m²)"),
        dcc.RangeSlider(
            id="size-slider",
            min=SIZE_MIN,
            max=SIZE_MAX,
            value=[SIZE_MIN, SIZE_MAX],
            step=5,  # 5 square meter steps
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode="mouseup",
            className="mb-4",
        ),
        
        # Maximum distance filters
        html.H6("Max Distances (km)"),
        
        # City distance slider
        html.Label("City Center"),
        dcc.Slider(
            id="city-dist-slider", min=0, max=30, value=30, step=1,
            marks={0: "0", 10: "10", 20: "20", 30: "30"}, updatemode="mouseup"
        ),
        html.Br(),
        
        # Park distance slider
        html.Label("Park"),
        dcc.Slider(
            id="park-dist-slider", min=0, max=10, value=10, step=0.5,
            marks={0: "0", 5: "5", 10: "10"}, updatemode="mouseup"
        ),
        html.Br(),
        
        # School distance slider
        html.Label("School"),
        dcc.Slider(
            id="school-dist-slider", min=0, max=10, value=10, step=0.5,
            marks={0: "0", 5: "5", 10: "10"}, updatemode="mouseup"
        ),
        html.Br(),
        
        # Transport distance slider
        html.Label("Transport"),
        dcc.Slider(
            id="transport-dist-slider", min=0, max=10, value=10, step=0.5,
            marks={0: "0", 5: "5", 10: "10"}, updatemode="mouseup"
        ),
        
        # Reset button
        html.Hr(),
        dbc.Button("Reset Filters", id="reset-filters", color="secondary", 
                  outline=True, className="w-100 mt-2"),
        html.Hr(),
        
        # Statistics display (updated by callback)
        html.Div(id="filter-stats"),
    ],
    className="bg-light p-4",  # CSS classes: light background, padding
    style={"height": "100vh", "overflowY": "auto"},  # Full height, scrollable
)

# ============================================================================
# KPI CARDS (top metrics display)
# ============================================================================
def kpi_card(title, value_id, color_class):
    """
    Create a KPI (Key Performance Indicator) card.
    
    Example: Small box showing "Total Properties: 1,234"
    """
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),  # Title (small, muted)
            html.H3(id=value_id, className=f"mb-0 {color_class}")  # Value (large, colored)
        ]),
        className="shadow-sm",  # Subtle shadow
    )

# Create KPI row with 4 cards
kpis = dbc.Row(
    [
        dbc.Col(kpi_card("Total Properties", "kpi-total", "text-primary"), md=3),
        dbc.Col(kpi_card("Average Price", "kpi-avg-price", "text-success"), md=3),
        dbc.Col(kpi_card("Average Price / m²", "kpi-avg-psqm", "text-info"), md=3),
        dbc.Col(kpi_card("Avg City Distance", "kpi-avg-city", "text-warning"), md=3),
    ],
    className="g-3 mb-3",  # Grid spacing
)

# ============================================================================
# TABS (navigation between different views)
# ============================================================================
tabs = dbc.Tabs(
    [
        dbc.Tab(label="🏠 Overview", tab_id="tab-overview"),
        dbc.Tab(label="📍 Location", tab_id="tab-location"),
        dbc.Tab(label="🎓 Education", tab_id="tab-education"),
        dbc.Tab(label="🚆 Transport", tab_id="tab-transport"),
        dbc.Tab(label="🏞️ Parks", tab_id="tab-parks"),
        dbc.Tab(label="📋 Data", tab_id="tab-data"),
    ],
    id="tabs",  # Component ID
    active_tab="tab-overview",  # Default active tab
    className="mb-3",
)

# ============================================================================
# TAB CONTENT (each tab has different charts)
# ============================================================================
# Tab 1: Overview
tab_overview = html.Div(
    [
        # Row 1: Histogram + Scatter
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="ov-hist", figure=empty_fig("Loading...")), md=6),
                dbc.Col(dcc.Graph(id="ov-scatter", figure=empty_fig("Loading...")), md=6),
            ],
            className="g-3",
        ),
        html.Div(style={"height": "10px"}),  # Spacer
        
        # Row 2: Correlation heatmap (full width)
        dbc.Row([dbc.Col(dcc.Graph(id="ov-corr", figure=empty_fig("Loading...")), md=12)], className="g-3"),
        html.Div(style={"height": "10px"}),  # Spacer
        
        # Row 3: Box plot + Map
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="ov-box", figure=empty_fig("Loading...")), md=6),
                dbc.Col(dcc.Graph(id="ov-map", figure=empty_fig("Loading...")), md=6),
            ],
            className="g-3",
        ),
    ]
)

# Tab 2: Location (city center impact)
tab_location = html.Div(
    [
        dbc.Row([dbc.Col(dcc.Graph(id="loc-scatter", figure=empty_fig("Loading...")), md=12)], className="g-3"),
        html.Div(style={"height": "10px"}),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="loc-bar", figure=empty_fig("Loading...")), md=6),
                dbc.Col(dcc.Graph(id="loc-trend", figure=empty_fig("Loading...")), md=6),
            ],
            className="g-3",
        ),
    ]
)

# Tab 3: Education (school impact)
tab_education = html.Div(
    [
        dbc.Row([dbc.Col(dcc.Graph(id="edu-scatter", figure=empty_fig("Loading...")), md=12)], className="g-3"),
        html.Div(style={"height": "10px"}),
        dbc.Row([dbc.Col(dcc.Graph(id="edu-bar", figure=empty_fig("Loading...")), md=12)], className="g-3"),
    ]
)

# Tab 4: Transport (bus/Luas impact)
tab_transport = html.Div(
    [
        dbc.Row([dbc.Col(dcc.Graph(id="tr-scatter", figure=empty_fig("Loading...")), md=12)], className="g-3"),
        html.Div(style={"height": "10px"}),
        dbc.Row([dbc.Col(dcc.Graph(id="tr-bar", figure=empty_fig("Loading...")), md=12)], className="g-3"),
    ]
)

# Tab 5: Parks (park impact)
tab_parks = html.Div(
    [
        dbc.Row([dbc.Col(dcc.Graph(id="pk-scatter", figure=empty_fig("Loading...")), md=12)], className="g-3"),
        html.Div(style={"height": "10px"}),
        dbc.Row([dbc.Col(dcc.Graph(id="pk-bar", figure=empty_fig("Loading...")), md=12)], className="g-3"),
    ]
)

# Tab 6: Data (table view and export)
tab_data = html.Div(
    [
        dbc.Row(
            [
                # Left column: Export controls
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Export"),
                                dbc.Button("📥 Download CSV", id="btn-csv", color="primary"),
                                dcc.Download(id="download-csv"),  # Hidden download component
                                html.Hr(),
                                html.Div(id="data-summary", className="text-muted small"),
                            ]
                        )
                    ),
                    md=4,
                ),
                # Right column: Data table
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Data Explorer (filter/sort)"),
                                dash_table.DataTable(
                                    id="data-table",
                                    data=[],  # Empty initially (filled by callback)
                                    columns=[],  # Empty initially (filled by callback)
                                    page_size=25,  # Rows per page
                                    sort_action="native",  # Allow sorting
                                    filter_action="native",  # Allow filtering
                                    page_action="native",  # Pagination
                                    style_table={"overflowX": "auto"},  # Horizontal scroll
                                    style_cell={  # Cell styling
                                        "textAlign": "left", 
                                        "padding": "8px", 
                                        "fontSize": "13px",
                                        "minWidth": "110px", 
                                        "maxWidth": "350px"
                                    },
                                    style_header={  # Header styling
                                        "backgroundColor": "#f8f9fa", 
                                        "fontWeight": "bold"
                                    },
                                ),
                            ]
                        )
                    ),
                    md=8,
                ),
            ],
            className="g-3",
        )
    ]
)

# ============================================================================
# MAIN LAYOUT (combine all components)
# ============================================================================
app.layout = dbc.Container(
    [
        # Hidden storage for filtered row indexes (fast data passing)
        dcc.Store(id="filtered-idx"),
        
        # Main row: Sidebar + Content
        dbc.Row(
            [
                # Left column: Sidebar (3/12 width)
                dbc.Col(sidebar, md=3, className="p-0"),
                
                # Right column: Main content (9/12 width)
                dbc.Col(
                    [
                        # Title
                        html.H2("📊 Dublin Property Analysis Dashboard", className="mt-3"),
                        
                        # KPI cards
                        kpis,
                        
                        # Tabs navigation
                        tabs,
                        
                        # Tab content (all tabs stacked, only one visible at a time)
                        html.Div(
                            [
                                html.Div(tab_overview, id="wrap-overview"),
                                html.Div(tab_location, id="wrap-location", style={"display": "none"}),
                                html.Div(tab_education, id="wrap-education", style={"display": "none"}),
                                html.Div(tab_transport, id="wrap-transport", style={"display": "none"}),
                                html.Div(tab_parks, id="wrap-parks", style={"display": "none"}),
                                html.Div(tab_data, id="wrap-data", style={"display": "none"}),
                            ]
                        ),
                    ],
                    md=9,
                ),
            ],
            className="g-0",  # No gutter (spacing) between columns
        ),
        
        # Bottom spacer
        html.Div(style={"height": "20px"}),
    ],
    fluid=True,  # Container uses full width
)

print("✅ UI layout created")


###############################################################################
# SECTION 13: DASH CALLBACKS (interactivity)
###############################################################################
# Callbacks connect user interactions (inputs) to chart updates (outputs)

print("Setting up callbacks...")

# ============================================================================
# CALLBACK 1: Tab visibility switching
# ============================================================================
@app.callback(
    # Outputs: Show/hide each tab content div
    Output("wrap-overview", "style"),
    Output("wrap-location", "style"),
    Output("wrap-education", "style"),
    Output("wrap-transport", "style"),
    Output("wrap-parks", "style"),
    Output("wrap-data", "style"),
    
    # Input: Which tab is active
    Input("tabs", "active_tab"),
)
def toggle_tab_visibility(active):
    """
    Show only the active tab, hide others.
    
    When user clicks a tab, Dash sends the tab_id to this function.
    We return style dictionaries: {} means show, {"display": "none"} means hide.
    """
    def vis(tab_id):
        # If this tab is active, return empty dict (show it)
        # Otherwise return dict with display:none (hide it)
        return {} if active == tab_id else {"display": "none"}
    
    # Return visibility for each tab
    return (
        vis("tab-overview"),
        vis("tab-location"),
        vis("tab-education"),
        vis("tab-transport"),
        vis("tab-parks"),
        vis("tab-data"),
    )


# ============================================================================
# CALLBACK 2: Filter data based on slider values
# ============================================================================
@app.callback(
    # Output: Store filtered row indexes
    Output("filtered-idx", "data"),
    
    # Inputs: All filter sliders (auto-trigger when any slider changes)
    Input("year-slider", "value"),
    Input("price-slider", "value"),
    Input("size-slider", "value"),
    Input("city-dist-slider", "value"),
    Input("park-dist-slider", "value"),
    Input("school-dist-slider", "value"),
    Input("transport-dist-slider", "value"),
)
def update_filtered_idx_auto(year_range, price_range, size_range, max_city, max_park, max_school, max_transport):
    """
    Apply all filters and store matching row indexes.
    
    This runs automatically whenever any slider changes.
    Returns list of row indexes that match ALL current filters.
    """
    # Start with all data
    f = df
    
    # Step 1: Core filters (year, price, city distance)
    mask = (
        f[YEAR_COL].between(year_range[0], year_range[1]) &  # Year within range
        f[PRICE_COL].between(price_range[0], price_range[1]) &  # Price within range
        (f["dist_to_city_center_km"] <= max_city)  # Within max city distance
    )
    
    # Step 2: Optional size filter
    if SIZE_COL in f.columns:
        mask = mask & f[SIZE_COL].between(size_range[0], size_range[1])
    
    # Step 3: Distance filters (treat missing distances as very large)
    # .fillna(9999): If distance is missing (NaN), use 9999 km (will fail <= max)
    if "dist_to_park_km" in f.columns:
        mask = mask & (f["dist_to_park_km"].fillna(9999) <= max_park)
    if "dist_to_school_km" in f.columns:
        mask = mask & (f["dist_to_school_km"].fillna(9999) <= max_school)
    if "dist_to_transport_km" in f.columns:
        mask = mask & (f["dist_to_transport_km"].fillna(9999) <= max_transport)
    
    # Step 4: Return indexes of rows that passed all filters
    # .index[mask]: Get row numbers where mask is True
    # .tolist(): Convert to Python list
    return df.index[mask].tolist()


# ============================================================================
# CALLBACK 3: Reset all filters
# ============================================================================
@app.callback(
    # Outputs: Reset all sliders to default values
    Output("year-slider", "value"),
    Output("price-slider", "value"),
    Output("size-slider", "value"),
    Output("city-dist-slider", "value"),
    Output("park-dist-slider", "value"),
    Output("school-dist-slider", "value"),
    Output("transport-dist-slider", "value"),
    
    # Input: Reset button click
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,  # Don't run when page first loads
)
def reset_controls(_):
    """
    Reset all filters to default values when reset button clicked.
    
    The underscore (_) parameter is the button click count (not used).
    """
    return (YEAR_UI_DEFAULT,  # Year range
            [PRICE_MIN, PRICE_MAX],  # Price range
            [SIZE_MIN, SIZE_MAX],  # Size range
            30, 10, 10, 10)  # Max distances


# ============================================================================
# CALLBACK 4: Update KPI cards
# ============================================================================
@app.callback(
    # Outputs: Update the 4 KPI values
    Output("kpi-total", "children"),
    Output("kpi-avg-price", "children"),
    Output("kpi-avg-psqm", "children"),
    Output("kpi-avg-city", "children"),
    
    # Input: Filtered row indexes
    Input("filtered-idx", "data"),
)
def update_kpis(idx_list):
    """
    Calculate and display KPI metrics for current filtered data.
    """
    # Get filtered dataframe from indexes
    f = df_from_idx(idx_list)
    
    # If no data, return zeros/N/A
    if f.empty:
        return "0", "€0", "N/A", "0.0 km"
    
    # Calculate metrics
    avg_price = f[PRICE_COL].mean()  # Average price
    # Average price per sqm (if column exists)
    avg_psqm = f["price_per_sqm"].mean() if "price_per_sqm" in f.columns else np.nan
    avg_city = f["dist_to_city_center_km"].mean()  # Average city distance
    
    # Format and return
    return (
        f"{len(f):,}",  # Total count with thousand separators
        f"€{avg_price:,.0f}",  # Average price formatted
        f"€{avg_psqm:,.0f}" if np.isfinite(avg_psqm) else "N/A",  # Price per sqm or N/A
        f"{avg_city:.1f} km",  # Average distance with 1 decimal
    )


# ============================================================================
# CALLBACK 5: Update sidebar statistics
# ============================================================================
@app.callback(
    Output("filter-stats", "children"),  # Update sidebar stats div
    Input("filtered-idx", "data"),  # When filtered data changes
)
def sidebar_stats(idx_list):
    """
    Show quick statistics in sidebar for current filtered data.
    """
    # Get filtered dataframe
    f = df_from_idx(idx_list)
    
    # If no data, show warning
    if f.empty:
        return dbc.Alert("No data for current filters", color="warning")
    
    # Calculate average size
    avg_size = f[SIZE_COL].mean() if SIZE_COL in f.columns else np.nan
    
    # Create stats display
    return html.Div(
        [
            html.H6("Current Results", className="mt-2"),
            html.Div(f"Properties: {len(f):,}"),
            html.Div(f"Avg Price: €{f[PRICE_COL].mean():,.0f}"),
            html.Div(f"Avg Size: {avg_size:.0f} m²" if np.isfinite(avg_size) else "Avg Size: N/A"),
            html.Div(f"Avg City Dist: {f['dist_to_city_center_km'].mean():.1f} km"),
        ],
        className="small text-muted",  # Small, muted text
    )


# ============================================================================
# CALLBACK 6: Update Overview tab charts
# ============================================================================
@app.callback(
    # Outputs: Update all 5 charts in Overview tab
    Output("ov-hist", "figure"),
    Output("ov-scatter", "figure"),
    Output("ov-corr", "figure"),
    Output("ov-box", "figure"),
    Output("ov-map", "figure"),
    
    # Inputs: Tab activity, filtered data, selected metric
    Input("tabs", "active_tab"),
    Input("filtered-idx", "data"),
    Input("metric", "value"),
)
def update_overview_figs(active_tab, idx_list, metric):
    """
    Build all charts for Overview tab.
    Only runs when Overview tab is active.
    """
    # If not Overview tab, don't update (save computation)
    if active_tab != "tab-overview":
        raise PreventUpdate  # Dash special: stop callback execution
    
    # Define function to build charts (for caching)
    def build():
        # Get filtered data
        f = df_from_idx(idx_list)
        
        # If no data, return empty figures
        if f.empty:
            e = empty_fig("No data for selected filters")
            return e, e, e, e, e
        
        # Clean data: remove infinite values
        safe = f.replace([np.inf, -np.inf], np.nan).copy()
        
        # Check if selected metric exists
        if metric not in safe.columns:
            e = empty_fig(f"Metric not found: {metric}")
            return e, e, e, e, e
        
        # Remove rows with missing metric or price
        safe = safe.dropna(subset=[metric, PRICE_COL])
        if safe.empty:
            e = empty_fig("No rows after cleaning (metric/price missing)")
            return e, e, e, e, e
        
        # Get user-friendly metric label
        mlabel = metric_label(metric)
        
        # --------------------------------------------------------------------
        # CHART 1: Histogram (distribution of prices)
        # --------------------------------------------------------------------
        try:
            fig1 = px.histogram(
                safe,
                x=metric,  # What to show on x-axis
                nbins=45,  # Number of bars
                title=f"Distribution: {mlabel}",
                labels={metric: mlabel},  # Axis label
            )
            fig1 = fig_layout(fig1)  # Apply consistent layout
        except Exception as ex:
            fig1 = empty_fig(f"Histogram error: {ex}")
        
        # --------------------------------------------------------------------
        # CHART 2: Scatter plot (price vs size or distance)
        # --------------------------------------------------------------------
        try:
            # Sample data for performance
            sc = safe.dropna(subset=[metric])
            sc = sample_df(sc, 15000)  # Use only 15,000 points
            
            # Choose x-axis: size if available, otherwise city distance
            xcol = SIZE_COL if SIZE_COL in sc.columns else "dist_to_city_center_km"
            
            # Color points by city distance if available
            color_col = "dist_to_city_center_km" if "dist_to_city_center_km" in sc.columns else None
            
            fig2 = px.scatter(
                sc,
                x=xcol,
                y=metric,
                color=color_col,
                title=f"{mlabel} vs {'Property Size' if xcol==SIZE_COL else 'City Distance'} (sampled)",
                labels={
                    xcol: "Property Size (m²)" if xcol == SIZE_COL else "City Distance (km)",
                    metric: mlabel
                },
                hover_data=[ADDRESS_COL] if ADDRESS_COL in sc.columns else None,
                render_mode="webgl",  # Fast WebGL rendering
            )
            fig2 = fig_layout(fig2)
        except Exception as ex:
            fig2 = empty_fig(f"Scatter error: {ex}")
        
        # --------------------------------------------------------------------
        # CHART 3: Correlation heatmap
        # --------------------------------------------------------------------
        try:
            fig_corr = build_correlation_heatmap(safe)
        except Exception as ex:
            fig_corr = empty_fig(f"Correlation heatmap error: {ex}")
        
        # --------------------------------------------------------------------
        # CHART 4: Box plot by city proximity
        # --------------------------------------------------------------------
        try:
            if "city_proximity" in safe.columns:
                # Prepare data
                bdf = safe.dropna(subset=["city_proximity", metric]).copy()
                bdf["city_proximity"] = normalize_category_series(bdf["city_proximity"])
                
                # Get categories actually present in filtered data
                x_order = present_categories(bdf, "city_proximity", CITY_ORDER)
                
                # Create box plot
                fig3 = px.box(
                    bdf,
                    x="city_proximity",
                    y=metric,
                    color="city_proximity",
                    title=f"{mlabel} by City Proximity",
                    labels={"city_proximity": "City Proximity", metric: mlabel},
                )
                
                # Force correct x-axis order
                fig3 = apply_x_category_order(fig3, x_order)
                fig3 = fig_layout(fig3)
            else:
                fig3 = empty_fig("city_proximity column not available")
        except Exception as ex:
            fig3 = empty_fig(f"Box plot error: {ex}")
        
        # --------------------------------------------------------------------
        # CHART 5: Map of property locations
        # --------------------------------------------------------------------
        try:
            # Get properties with location data
            mdf = safe.dropna(subset=["lat", "lon", PRICE_COL]).copy()
            mdf = sample_df(mdf, 1200)  # Sample for performance
            
            if mdf.empty:
                fig4 = empty_fig("No map points available (missing lat/lon)")
            else:
                fig4 = px.scatter_mapbox(
                    mdf,
                    lat="lat",
                    lon="lon",
                    color=PRICE_COL,  # Color by price
                    hover_name=ADDRESS_COL if ADDRESS_COL in mdf.columns else None,
                    zoom=10,  # Initial zoom level
                    height=450,
                    title="Map: Property Locations (sampled)",
                )
                # Customize markers
                fig4.update_traces(marker=dict(size=5, opacity=0.65))
                # Use OpenStreetMap tiles
                fig4.update_layout(mapbox_style="open-street-map")
                fig4 = fig_layout(fig4, height=450)
        except Exception as ex:
            fig4 = empty_fig(f"Map error: {ex}")
        
        return fig1, fig2, fig_corr, fig3, fig4
    
    # Use cache if available
    return cached_or_build("figs", "tab-overview", metric, idx_list, build)


# ============================================================================
# HELPER: Build proximity charts (used by multiple tabs)
# ============================================================================
def build_proximity_figs(idx_list, metric, dist_col, cat_col, title_prefix):
    """
    Generic function to build two charts for any proximity feature.
    
    Creates:
      1. Scatter plot: metric vs numeric distance
      2. Bar chart: average metric by distance bucket
    
    Used by Education, Transport, and Parks tabs.
    """
    # Get filtered data
    f = df_from_idx(idx_list)
    
    # Check for empty data
    if f.empty:
        e = empty_fig("No data for selected filters")
        return e, e
    
    # Check if distance data exists
    if dist_col not in f.columns or f[dist_col].dropna().empty:
        e = empty_fig(f"{title_prefix} distance data not available")
        return e, e
    
    # Get metric label
    mlabel = metric_label(metric)
    
    # Clean data
    safe = f.dropna(subset=[metric, dist_col]).copy()
    if safe.empty:
        e = empty_fig("Not enough data after cleaning")
        return e, e
    
    # Get correct category order list
    full_order = CATEGORY_ORDERS.get(cat_col, None)
    
    # Clean category strings
    if cat_col in safe.columns:
        safe[cat_col] = normalize_category_series(safe[cat_col])
    
    # --------------------------------------------------------------------
    # CHART 1: Scatter plot (sampled)
    # --------------------------------------------------------------------
    sc = sample_df(safe, 15000)
    fig1 = px.scatter(
        sc,
        x=dist_col,
        y=metric,
        color=cat_col if cat_col in sc.columns else None,
        title=f"{mlabel} vs Distance to {title_prefix} (sampled)",
        labels={dist_col: f"Distance to {title_prefix} (km)", metric: mlabel},
        render_mode="webgl",
    )
    fig1 = fig_layout(fig1)
    
    # --------------------------------------------------------------------
    # CHART 2: Bar chart (average by bucket)
    # --------------------------------------------------------------------
    # Group by category and calculate average metric
    by_cat = safe.groupby(cat_col, observed=True)[metric].mean().reset_index()
    
    # Sort dataframe in correct bucket order
    by_cat = order_dataframe_by_category(by_cat, cat_col, full_order)
    
    # Get x-axis order list
    x_order = present_categories(by_cat, cat_col, full_order) if full_order else None
    
    # Create bar chart
    fig2 = px.bar(
        by_cat,
        x=cat_col,
        y=metric,
        color=cat_col,
        title=f"Average {mlabel} by {title_prefix} Proximity",
        labels={cat_col: f"{title_prefix} Proximity", metric: f"Avg {mlabel}"},
    )
    
    # Force x-axis order
    fig2 = apply_x_category_order(fig2, x_order)
    fig2 = fig_layout(fig2, height=420)
    
    return fig1, fig2


# ============================================================================
# CALLBACK 7: Update Location tab charts
# ============================================================================
@app.callback(
    Output("loc-scatter", "figure"),
    Output("loc-bar", "figure"),
    Output("loc-trend", "figure"),
    Input("tabs", "active_tab"),
    Input("filtered-idx", "data"),
    Input("metric", "value"),
)
def update_location_figs(active_tab, idx_list, metric):
    """
    Build charts for Location tab (city center impact).
    """
    if active_tab != "tab-location":
        raise PreventUpdate
    
    def build():
        # Get filtered data
        f = df_from_idx(idx_list)
        
        if f.empty:
            e = empty_fig("No data for selected filters")
            return e, e, e
        
        # Get metric label
        mlabel = metric_label(metric)
        
        # Clean data
        safe = f.dropna(subset=[metric, "dist_to_city_center_km"]).copy()
        if safe.empty:
            e = empty_fig("Not enough data after cleaning")
            return e, e, e
        
        # Clean categories
        safe["city_proximity"] = normalize_category_series(safe["city_proximity"])
        
        # Sample for scatter plot
        sc = sample_df(safe, 15000)
        
        # CHART 1: Scatter plot
        fig1 = px.scatter(
            sc,
            x="dist_to_city_center_km",
            y=metric,
            color="city_proximity",
            title=f"{mlabel} vs Distance to City Center (sampled)",
            labels={"dist_to_city_center_km": "City Distance (km)", metric: mlabel},
            render_mode="webgl",
        )
        fig1 = fig_layout(fig1)
        
        # CHART 2: Bar chart (average by city proximity)
        by_cat = safe.groupby("city_proximity", observed=True)[metric].mean().reset_index()
        by_cat = order_dataframe_by_category(by_cat, "city_proximity", CITY_ORDER)
        x_order = present_categories(by_cat, "city_proximity", CITY_ORDER)
        
        fig2 = px.bar(
            by_cat,
            x="city_proximity",
            y=metric,
            color="city_proximity",
            title=f"Average {mlabel} by City Proximity",
            labels={"city_proximity": "City Proximity", metric: f"Avg {mlabel}"},
        )
        fig2 = apply_x_category_order(fig2, x_order)
        fig2 = fig_layout(fig2, height=420)
        
        # CHART 3: Trend by year
        if YEAR_COL in safe.columns and not safe[YEAR_COL].dropna().empty:
            # Group by year, calculate average metric
            trend = safe.dropna(subset=[YEAR_COL]).groupby(YEAR_COL, observed=True)[metric].mean().reset_index()
            fig3 = px.line(
                trend,
                x=YEAR_COL,
                y=metric,
                markers=True,
                title=f"Trend: Avg {mlabel} by Year",
                labels={YEAR_COL: "Sale Year", metric: f"Avg {mlabel}"},
            )
            fig3 = fig_layout(fig3, height=420)
        else:
            fig3 = empty_fig("Trend: sale_year not available")
        
        return fig1, fig2, fig3
    
    # Use cache
    return cached_or_build("figs", "tab-location", metric, idx_list, build)


# ============================================================================
# CALLBACK 8: Update Education tab charts
# ============================================================================
@app.callback(
    Output("edu-scatter", "figure"),
    Output("edu-bar", "figure"),
    Input("tabs", "active_tab"),
    Input("filtered-idx", "data"),
    Input("metric", "value"),
)
def update_education_figs(active_tab, idx_list, metric):
    """
    Build charts for Education tab (school impact).
    """
    if active_tab != "tab-education":
        raise PreventUpdate
    
    # Use proximity helper with school parameters
    return cached_or_build(
        "figs", "tab-education", metric, idx_list,
        lambda: build_proximity_figs(idx_list, metric, 
                                     "dist_to_school_km", 
                                     "school_proximity", 
                                     "Schools")
    )


# ============================================================================
# CALLBACK 9: Update Transport tab charts
# ============================================================================
@app.callback(
    Output("tr-scatter", "figure"),
    Output("tr-bar", "figure"),
    Input("tabs", "active_tab"),
    Input("filtered-idx", "data"),
    Input("metric", "value"),
)
def update_transport_figs(active_tab, idx_list, metric):
    """
    Build charts for Transport tab (bus/Luas impact).
    """
    if active_tab != "tab-transport":
        raise PreventUpdate
    
    # Use proximity helper with transport parameters
    return cached_or_build(
        "figs", "tab-transport", metric, idx_list,
        lambda: build_proximity_figs(idx_list, metric,
                                     "dist_to_transport_km",
                                     "transport_proximity",
                                     "Transport")
    )


# ============================================================================
# CALLBACK 10: Update Parks tab charts
# ============================================================================
@app.callback(
    Output("pk-scatter", "figure"),
    Output("pk-bar", "figure"),
    Input("tabs", "active_tab"),
    Input("filtered-idx", "data"),
    Input("metric", "value"),
)
def update_parks_figs(active_tab, idx_list, metric):
    """
    Build charts for Parks tab (park impact).
    """
    if active_tab != "tab-parks":
        raise PreventUpdate
    
    # Use proximity helper with park parameters
    return cached_or_build(
        "figs", "tab-parks", metric, idx_list,
        lambda: build_proximity_figs(idx_list, metric,
                                     "dist_to_park_km",
                                     "park_proximity",
                                     "Parks")
    )


# ============================================================================
# CALLBACK 11: Update Data tab (table and summary)
# ============================================================================
@app.callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Output("data-summary", "children"),
    Input("tabs", "active_tab"),
    Input("filtered-idx", "data"),
)
def update_data_tab(active_tab, idx_list):
    """
    Update Data tab with filtered data table.
    """
    # Only run if Data tab is active
    if active_tab != "tab-data":
        raise PreventUpdate
    
    # Get filtered data
    f = df_from_idx(idx_list)
    
    # If no data, return empty
    if f.empty:
        return [], [], "No data for current filters"
    
    # Preferred column order for display
    preferred = [
        YEAR_COL,
        ADDRESS_COL,
        PRICE_COL,
        SIZE_COL,
        "price_per_sqm",
        "dist_to_city_center_km",
        "dist_to_park_km",
        "dist_to_school_km",
        "dist_to_transport_km",
        "city_proximity",
        "park_proximity",
        "school_proximity",
        "transport_proximity",
    ]
    
    # Keep only columns that exist
    cols = [c for c in preferred if c in f.columns]
    
    # Take first 800 rows (for performance)
    view = f[cols].head(800).copy()
    
    # Convert to dictionary format for DataTable
    data = view.to_dict("records")
    
    # Create column definitions for DataTable
    columns = [{"name": c, "id": c} for c in cols]
    
    # Create summary text
    summary = f"Showing {len(view):,} rows (of {len(f):,} filtered rows). Use Download for full data."
    
    return data, columns, summary


# ============================================================================
# CALLBACK 12: Download CSV
# ============================================================================
@app.callback(
    Output("download-csv", "data"),  # Trigger file download
    Input("btn-csv", "n_clicks"),    # When download button clicked
    State("filtered-idx", "data"),   # Current filtered data indexes
    prevent_initial_call=True,       # Don't run on page load
)
def download_csv(n, idx_list):
    """
    Export current filtered data to CSV file.
    """
    # Get filtered data
    f = df_from_idx(idx_list)
    
    # Trigger download
    # dcc.send_data_frame creates and sends CSV file
    return dcc.send_data_frame(
        f.to_csv,  # Function to convert dataframe to CSV
        "dublin_properties_filtered_2020_2025.csv",  # Filename
        index=False  # Don't include row numbers
    )


###############################################################################
# SECTION 14: RUN THE APPLICATION
###############################################################################
if __name__ == "__main__":
    # Print startup information
    print("\n" + "=" * 60)
    print("Dublin Property Dashboard (FIXED: No KeyError + Ordered Buckets)")
    print("=" * 60)
    print(f"Loaded rows (2020–2025 only): {len(df):,}")
    print(f"Year range present: {int(df[YEAR_COL].min())} - {int(df[YEAR_COL].max())}")
    print(f"Price range: €{df[PRICE_COL].min():,.0f} - €{df[PRICE_COL].max():,.0f}")
    print("=" * 60)
    print("Open: http://localhost:8050")
    print("=" * 60)
    print("\nStarting server... (Press Ctrl+C to stop)")
    
    # Run the Dash application
    # debug=True: Shows detailed errors and auto-reloads on code changes
    # debug=False: Use for production (faster, no error details)
    app.run(debug=False, port=8050)
