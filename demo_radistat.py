import SimpleITK as sitk

from radistat import radistat

if __name__ == "__main__":
    # Load in data
    vol = sitk.ReadImage("example_data/ex1_vol.mha")
    vol_mask = sitk.ReadImage("example_data/ex1_mask.mha")
    vol_featmap = sitk.ReadImage("example_data/ex1_feature_map.mha")

    # Convert data to numpy structures
    vol = sitk.GetArrayFromImage(vol)
    vol_mask = sitk.GetArrayFromImage(vol_mask)
    vol_featmap = sitk.GetArrayFromImage(vol_featmap)

    # We need to rotate this data. SITK puts the depth axis first, so make that last
    vol = vol.transpose(1, 2, 0)
    vol_mask = vol_mask.transpose(1, 2, 0)
    vol_featmap = vol_featmap.transpose(1, 2, 0)

    ## 2D

    # Pick one particular slice and run it through RADISTAT
    SLICE = 66
    img = vol[:, :, SLICE]
    mask = vol_mask[:, :, SLICE]
    featmap = vol_featmap[:, :, SLICE]

    radistat(img, mask, featmap)

    # TODO visualize results
