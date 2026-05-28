"""
Geospatial Research Database Models.
=====================================
Tabel terpisah untuk modul Geospatial AI Research.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, Boolean, DateTime,
    ForeignKey, JSON,
)
from sqlalchemy.orm import relationship
from app.database.models import Base


class GeospatialProject(Base):
    """Project untuk research banjir per user."""
    __tablename__ = "geospatial_projects"

    project_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    project_name = Column(String(255), nullable=False)
    research_goal = Column(String(100))  # flood_classification / susceptibility / extent_mapping
    study_area = Column(String(255))      # nama area, e.g., "Jakarta Pusat"
    aoi_geojson = Column(JSON)            # bounding box atau polygon AOI
    target_classes = Column(JSON)         # ["flooded", "non_flooded"] atau ["high_risk", ...]
    crs = Column(String(50))              # primary CRS untuk project
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    raster_files = relationship("RasterFile", back_populates="project", cascade="all, delete-orphan")
    model_runs = relationship("FloodModelRun", back_populates="project", cascade="all, delete-orphan")


class RasterFile(Base):
    """Single GeoTIFF file dalam project."""
    __tablename__ = "raster_files"

    raster_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("geospatial_projects.project_id", ondelete="CASCADE"), nullable=False)

    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer)
    raster_role = Column(String(50))      # "input_imagery" / "ground_truth" / "before" / "after"

    # Metadata dari GeoTIFF
    crs = Column(String(50))
    crs_epsg = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    band_count = Column(Integer)
    dtype = Column(String(20))
    resolution_x = Column(Float)
    resolution_y = Column(Float)
    nodata_value = Column(Float)

    # Bounding box
    bbox_min_x = Column(Float)
    bbox_min_y = Column(Float)
    bbox_max_x = Column(Float)
    bbox_max_y = Column(Float)

    # Tags / source
    source = Column(String(100))           # "sentinel-1" / "sentinel-2" / "landsat" / "dem"
    acquisition_date = Column(DateTime)
    metadata_json = Column(JSON)           # raw raster metadata

    uploaded_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("GeospatialProject", back_populates="raster_files")
    bands = relationship("RasterBand", back_populates="raster", cascade="all, delete-orphan")


class RasterBand(Base):
    """Per-band statistics dari raster."""
    __tablename__ = "raster_bands"

    band_id = Column(Integer, primary_key=True, index=True)
    raster_id = Column(Integer, ForeignKey("raster_files.raster_id", ondelete="CASCADE"), nullable=False)

    band_index = Column(Integer)            # 1-based
    band_name = Column(String(100))         # "red", "nir", "VV", dll
    min_value = Column(Float)
    max_value = Column(Float)
    mean_value = Column(Float)
    std_value = Column(Float)
    n_valid_pixels = Column(Integer)
    n_nodata_pixels = Column(Integer)
    nodata_percentage = Column(Float)

    raster = relationship("RasterFile", back_populates="bands")


class FloodModelRun(Base):
    """Hasil training model klasifikasi banjir."""
    __tablename__ = "flood_model_runs"

    run_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("geospatial_projects.project_id", ondelete="CASCADE"), nullable=False)

    model_name = Column(String(100))            # "random_forest", "xgboost", "mndwi_threshold"
    model_type = Column(String(50))             # "supervised" / "threshold" / "change_detection"
    features_used = Column(JSON)                # ["ndwi", "mndwi", "vv", "elevation", "slope"]
    target_variable = Column(String(100))       # "flood_label" atau null untuk threshold
    hyperparameters = Column(JSON)

    # Evaluation metrics
    metrics_json = Column(JSON)                  # {accuracy, precision, recall, f1, iou, kappa, ...}
    confusion_matrix = Column(JSON)
    feature_importance = Column(JSON)

    # Method monitor entries
    method_monitor_log = Column(JSON)            # full reasoning trace

    # Output paths
    output_map_path = Column(String(500))        # GeoTIFF flood mask
    output_probability_path = Column(String(500))
    output_geojson_path = Column(String(500))    # vector polygon hasil
    output_csv_path = Column(String(500))

    # Summary stats
    flooded_pixel_count = Column(Integer)
    flooded_area_km2 = Column(Float)
    flooded_percentage = Column(Float)

    status = Column(String(50), default="pending")  # pending / running / completed / failed
    error_message = Column(Text)

    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("GeospatialProject", back_populates="model_runs")
