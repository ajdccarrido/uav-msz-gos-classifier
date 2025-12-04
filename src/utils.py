import rasterio
import geopandas as gpd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

def write_geotiff(path, array, meta):
    meta2 = meta.copy()
    meta2.update(
        dtype="uint8",
        count=1,
        nodata=0
    )

    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(array.astype("uint8"), 1)

def validate_img(classified_img_path, validation_points_path, ref_col="class_id"):
    """
    Performs accuracy assessment on a classified raster using validation points.
    Automatically prints confusion matrix, OA, UA, PA, and Kappa.
    
    Parameters:
        classified_img_path (str): Path to classified raster.
        validation_points_path (str): Path to the validation points vector file.
        ref_col (str): Column name containing true/reference class values.
        
    Returns:
        dict: accuracy metrics and confusion matrix.
    """

    # --- 1. Load validation points from file ---
    validation_points_gdf = gpd.read_file(validation_points_path)

    # --- 2. Load classified raster ---
    with rasterio.open(classified_img_path) as src:
        classified = src.read(1)
        affine = src.transform
        nodata = src.nodata

    # --- 3. Extract predicted class at each validation point ---
    rows, cols = rasterio.transform.rowcol(
        affine,
        validation_points_gdf.geometry.x,
        validation_points_gdf.geometry.y
    )

    predicted = []
    for r, c in zip(rows, cols):
        try:
            val = classified[r, c]
            if nodata is not None and val == nodata:
                predicted.append(np.nan)
            else:
                predicted.append(val)
        except:
            predicted.append(np.nan)

    validation_points_gdf["predicted"] = predicted

    # Drop invalid predictions
    df = validation_points_gdf.dropna(subset=["predicted"])

    y_true = df[ref_col].astype(int).to_numpy()
    y_pred = df["predicted"].astype(int).to_numpy()

    # --- 4. Compute confusion matrix ---
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- 5. Accuracy metrics ---
    overall_accuracy = np.trace(cm) / cm.sum()
    users_accuracy = np.diag(cm) / cm.sum(axis=1)
    producers_accuracy = np.diag(cm) / cm.sum(axis=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # --- 6. Print Results ---
    print("\n================ ACCURACY ASSESSMENT ================\n")

    print("Confusion Matrix (rows = reference, cols = predicted):")
    print("Labels:", labels)
    print(cm)
    print("\n----------------------------------------------------")

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")

    print("----------------------------------------------------")

    print("User's Accuracy (per class):")
    for label, ua in zip(labels, users_accuracy):
        print(f"  Class {label}: {ua:.4f}")

    print("----------------------------------------------------")

    print("Producer's Accuracy (per class):")
    for label, pa in zip(labels, producers_accuracy):
        print(f"  Class {label}: {pa:.4f}")

    print("====================================================\n")

    # --- 7. Return Results ---
    return {
        "overall_accuracy": overall_accuracy,
        "kappa": kappa,
        "confusion_matrix": cm,
        "users_accuracy": users_accuracy,
        "producers_accuracy": producers_accuracy,
        "labels": labels
    }
