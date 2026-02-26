
import sqlite3
import numpy as np
import io
from PIL import Image
import math
import shapely.geometry as sg

class StaticMapPlotter:
    """A class to handle offline map generation from MBTiles."""
    
    def __init__(self, mbtiles_path: str):
        self.mbtiles_path = mbtiles_path

    def deg2num(self, lat_deg: float, lon_deg: float, zoom: int):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    def num2deg(self, xtile: int, ytile: int, zoom: int):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)

    def get_auto_params(self, data):
        """Calculate bbox and optimal zoom level from dataframe/series data."""
        geoms = []
        if hasattr(data, '_data'):
            if hasattr(data, 'columns'): # DataFrame
                col = 'geometry' if 'geometry' in data.columns else data.columns[0]
                geoms = data._data[col]
            else: # Series
                geoms = data._data
        
        # Get all points for envelope
        lons, lats = [], []
        for g in geoms:
            if isinstance(g, sg.Point):
                lons.append(g.x); lats.append(g.y)
            elif hasattr(g, 'bounds'):
                b = g.bounds
                lons.extend([b[0], b[2]]); lats.extend([b[1], b[3]])
        
        if not lons:
            # Default to some region if no data
            return (34.0, 35.0, 31.0, 32.0), 10

        # 1. Base range
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        
        # 2. Enforce minimum span (ensure map isn't a tiny sliver)
        # 0.02 degrees is roughly 2km - enough to see context at high zoom
        min_span = 0.03 
        if lon_range < min_span:
            mid_lon = (min_lon + max_lon) / 2
            min_lon, max_lon = mid_lon - min_span/2, mid_lon + min_span/2
            lon_range = min_span
            
        if lat_range < min_span:
            mid_lat = (min_lat + max_lat) / 2
            min_lat, max_lat = mid_lat - min_span/2, mid_lat + min_span/2
            lat_range = min_span

        # 3. Maintain healthy aspect ratio (min 1:3 or 3:1)
        # Prevents long thin vertical or horizontal strips
        if lon_range < lat_range * 0.4:
            mid_lon = (min_lon + max_lon) / 2
            lon_range = lat_range * 0.4
            min_lon, max_lon = mid_lon - lon_range/2, mid_lon + lon_range/2
        elif lat_range < lon_range * 0.4:
            mid_lat = (min_lat + max_lat) / 2
            lat_range = lon_range * 0.4
            min_lat, max_lat = mid_lat - lat_range/2, mid_lat + lat_range/2

        # 4. Final buffer (5% for breathing room)
        bbox = (min_lon - lon_range*0.05, max_lon + lon_range*0.05, 
                min_lat - lat_range*0.05, max_lat + lat_range*0.05)
        
        # Get max zoom from metadata
        max_zoom = 13
        try:
            conn = sqlite3.connect(self.mbtiles_path)
            cur = conn.cursor()
            cur.execute("SELECT value FROM metadata WHERE name='maxzoom'")
            row = cur.fetchone()
            if row: max_zoom = int(row[0])
            conn.close()
        except:
            pass

        # Optimal zoom calculation
        # Highest zoom that stays under 8192px total dimension (Safe High Res)
        MAX_DIM = 8192
        best_zoom = 5
        for z in range(max_zoom, 4, -1):
            try:
                x_min, y_min = self.deg2num(bbox[3], bbox[0], z)
                x_max, y_max = self.deg2num(bbox[2], bbox[1], z)
                w = (abs(x_max - x_min) + 1) * 256
                h = (abs(y_max - y_min) + 1) * 256
                if w <= MAX_DIM and h <= MAX_DIM:
                    best_zoom = z
                    break
            except:
                continue
        
        return bbox, best_zoom

    def get_static_map(self, bbox: tuple, zoom: int):
        MAX_DIM = 8192
        
        while zoom > 0:
            x_min, y_min = self.deg2num(bbox[3], bbox[0], zoom)
            x_max, y_max = self.deg2num(bbox[2], bbox[1], zoom)
            
            x_start, x_end = min(x_min, x_max), max(x_min, x_max)
            y_start, y_end = min(y_min, y_max), max(y_min, y_max)
            
            width = (x_end - x_start + 1) * 256
            height = (y_end - y_start + 1) * 256
            
            if width <= MAX_DIM and height <= MAX_DIM:
                break
            
            print(f"Warning: Requested map size ({width}x{height}) too large at zoom {zoom}. Downscaling...")
            zoom -= 1
        
        if zoom <= 0:
            raise ValueError("Bounding box too large to render even at zoom 0.")

        canvas = Image.new('RGB', (width, height))
        conn = sqlite3.connect(self.mbtiles_path)
        cur = conn.cursor()
        
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                tms_y = (1 << zoom) - 1 - y
                cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (zoom, x, tms_y))
                row = cur.fetchone()
                if row:
                    tile = Image.open(io.BytesIO(row[0]))
                    canvas.paste(tile, ((x - x_start) * 256, (y - y_start) * 256))
        conn.close()

        lat_tl, lon_tl = self.num2deg(x_start, y_start, zoom)
        lat_br, lon_br = self.num2deg(x_end + 1, y_end + 1, zoom)
        extent = [lon_tl, lon_br, lat_br, lat_tl]
        return canvas, extent

import os

def plot_static(data, mbtiles_path='liopandas/satelight_israel.mbtiles', bbox=None, zoom=None, output_path="offline_map.png", show_labels=True, show_city_labels=True, show_context=False):
    """Renders an offline static map image with liopandas data overlaid.
    If bbox and zoom are None, they are calculated automatically from the data.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
    except ImportError:
        raise ImportError("matplotlib is required for static plotting.")

    plotter = StaticMapPlotter(mbtiles_path)
    
    if bbox is None or zoom is None:
        auto_bbox, auto_zoom = plotter.get_auto_params(data)
        bbox = bbox if bbox is not None else auto_bbox
        zoom = zoom if zoom is not None else auto_zoom

    # 1. Fetch Main Map
    canvas, extent = plotter.get_static_map(bbox, zoom)

    if show_context:
        fig, (ax, ax_ctx) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Render Main Map
    ax.imshow(canvas, extent=extent)
    
    # Render Context Map if requested
    if show_context:
        # Whole region context (approx Israel bounds or zoomed out from data)
        ctx_zoom = max(0, zoom - 4)
        # Use full available map bounds or a reasonably larger area
        ctx_bbox = (34.0, 36.0, 29.5, 33.5) # Israel approx
        try:
            ctx_canvas, ctx_extent = plotter.get_static_map(ctx_bbox, ctx_zoom)
            ax_ctx.imshow(ctx_canvas, extent=ctx_extent)
            
            # Draw rectangle of the main map's extent on the context map
            rect = sg.box(extent[0], extent[2], extent[1], extent[3])
            rx, ry = rect.exterior.xy
            ax_ctx.plot(rx, ry, color='yellow', linewidth=3, path_effects=[path_effects.withStroke(linewidth=5, foreground='black')])
            ax_ctx.fill(rx, ry, color='yellow', alpha=0.2)
            ax_ctx.set_title("Regional Context", fontsize=14, fontweight='bold')
            ax_ctx.axis('off')
        except Exception as e:
            ax_ctx.text(0.5, 0.5, f"Context Map Unavailable: {e}", ha='center', va='center', fontsize=12, color='gray')
            ax_ctx.axis('off')

    # 2. Overlay City Labels from cities.csv if requested
    if show_city_labels:
        cities_file = os.path.join(os.path.dirname(__file__), 'cities.csv')
        if os.path.exists(cities_file):
            lon_min, lon_max, lat_min, lat_max = extent[0], extent[1], extent[2], extent[3]
            try:
                candidate_cities = []
                with open(cities_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            try:
                                lat = float(parts[1])
                                lon = float(parts[2])
                                # Check if city is within map extent
                                if lon_min <= lon <= lon_max and min(lat_min, lat_max) <= lat <= max(lat_min, lat_max):
                                    candidate_cities.append({'name': parts[0], 'lat': lat, 'lon': lon})
                            except ValueError: continue
                
                # Take top 10 in view
                top_cities = candidate_cities[:10]
                
                # Label collision avoidance
                occupied_rects = []
                # Approx conversion from font/char to degrees
                deg_w = (lon_max - lon_min)
                deg_h = abs(lat_max - lat_min)
                
                for city in top_cities:
                    name = city['name']
                    lon, lat = city['lon'], city['lat']
                    
                    # Approximate label box (width approx 0.015 of view width per char)
                    # Height approx 0.03 of view height
                    tw = len(name) * 0.012 * deg_w
                    th = 0.025 * deg_h
                    
                    # Label is slightly offset to the right: [lon + padding, lat - th/2, lon + padding + tw, lat + th/2]
                    padding = 0.005 * deg_w
                    rect = [lon + padding, lat - th/2, lon + padding + tw, lat + th/2]
                    
                    overlap = False
                    for r in occupied_rects:
                        # Standard AABB intersection
                        if not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3]):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Draw label
                        ax.text(lon, lat, f"  {name}", fontsize=10, color='white', fontweight='bold',
                                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')],
                                verticalalignment='center', zorder=8)
                        # Draw point ABOVE the label (higher zorder)
                        ax.plot(lon, lat, 'wo', markersize=5, markeredgecolor='black', alpha=0.9, zorder=10)
                        occupied_rects.append(rect)
                    else:
                        # If label overlaps, we might still want to draw the point? 
                        # The user asked for "display 10 top cities", implying we want to see them if possible.
                        # We'll skip the label but could potentially draw the point. 
                        # However, for "top 10" we usually want both.
                        pass

            except Exception as e:
                print(f"Warning: Could not process cities.csv: {e}")

    # 3. Overlay Data from DataFrame/Series
    geoms = []
    if hasattr(data, '_data'):
        if hasattr(data, 'columns'): # DataFrame
            col = 'geometry' if 'geometry' in data.columns else data.columns[0]
            geoms = data._data[col]
        else: # Series
            geoms = data._data

    for idx, geom in enumerate(geoms):
        label = None
        f_color = 'red'
        f_size = 8
        if hasattr(data, 'columns'):
            if show_labels:
                for col_name in ['city', 'name', 'label', 'location']:
                    if col_name in data.columns:
                        label = str(data._data[col_name][idx]); break
            if 'color' in data.columns: f_color = data._data['color'][idx]
            if 'size' in data.columns: f_size = float(data._data['size'][idx])

        if isinstance(geom, sg.Point):
            ax.plot(geom.x, geom.y, 'o', color=f_color, markersize=f_size, markeredgecolor='white', zorder=5)
            if label:
                ax.text(geom.x, geom.y, f"  {label}", fontsize=11, fontweight='bold', 
                        color='white', path_effects=[path_effects.withStroke(linewidth=3, foreground='black')],
                        verticalalignment='center', zorder=6)
        elif isinstance(geom, sg.LineString):
            x, y = geom.xy
            ax.plot(x, y, color=f_color, linewidth=f_size/4, zorder=4)
        elif isinstance(geom, sg.Polygon):
            x, y = geom.exterior.xy
            ax.fill(x, y, color=f_color, alpha=0.3, zorder=3)
            ax.plot(x, y, color=f_color, linewidth=1, zorder=3)
            if label:
                ax.text(geom.centroid.x, geom.centroid.y, label, fontsize=10, fontweight='bold',
                        color='white', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')],
                        horizontalalignment='center', zorder=6)
        elif hasattr(geom, 'geoms'): # Multi-geometries
            for g in geom.geoms:
                if isinstance(g, sg.LineString):
                    x, y = g.xy
                    ax.plot(x, y, color=f_color, linewidth=f_size/4, zorder=4)
                elif isinstance(g, sg.Polygon):
                    x, y = g.exterior.xy
                    ax.fill(x, y, color=f_color, alpha=0.3, zorder=3)
                    ax.plot(x, y, color=f_color, linewidth=1, zorder=3)
        elif hasattr(geom, 'centroid'):
            ax.plot(geom.centroid.x, geom.centroid.y, 's', color=f_color, markersize=f_size, zorder=5)

    ax.set_title("LioPandas Detailed View", fontsize=16, fontweight='bold')
    if not show_context: ax.axis('off') # Keep axis if showing comparative context
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    return output_path
