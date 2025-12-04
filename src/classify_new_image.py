# ------------------------- IMPORTS ------------------------- #
import joblib
import rasterio
from step1_load_data import load_raster_and_vectors
from step4_classify_raster import classify_tiled
from utils import write_geotiff, validate_img

# ------------------------- PATHS ------------------------- #

NEW_RASTER = "data/raw/QC_556_Stacked_3857.tif"       # <-- change this!
MODEL_PATH = "models/classifier_rf.joblib"  # saved RF model
VALIDATION_POINTS = "data/validation/556_validation.gpkg"
OUT_PATH   = "data/outputs/classified_new_image.tif"

# ------------------------- MAIN ------------------------- #

def main():
    # print("Loading trained classifier...")
    # clf = joblib.load(MODEL_PATH)

    # print("Loading new UAV raster...")
    # # note: we don't need vector/training for new images
    # arr, meta, _ = load_raster_and_vectors(NEW_RASTER, None)

    # print("Classifying new image (tiled)...")
    # rf_map = classify_tiled(NEW_RASTER, clf, meta)

    # print("Saving output...")
    # write_geotiff(OUT_PATH, rf_map, meta)

    # print("\nDONE! Classified image saved to:", OUT_PATH)

    print("Validating...")
    results = validate_img(OUT_PATH, VALIDATION_POINTS)

if __name__ == "__main__":
    main()
