import SimpleITK as sitk

from radistat import radistat

if __name__ == "__main__":
    # Load in data
    vol = sitk.ReadImage("example_data/ex1_vol.mha")
    volmask = sitk.ReadImage("example_data/ex1_mask.mha")
    vol_featmap = sitk.ReadImage("example_data/ex1_feature_map.mha")

    # Convert data to numpy structures
    vol = sitk.GetArrayFromImage(vol)
    volmask = sitk.GetArrayFromImage(volmask)
    vol_featmap = sitk.GetArrayFromImage(vol_featmap)

    # We need to rotate this data. SITK puts the depth axis first, so make that last
    vol = vol.transpose(1, 2, 0)
    volmask = volmask.transpose(1, 2, 0)
    vol_featmap = vol_featmap.transpose(1, 2, 0)

    ## 2D

    # Pick one particular slice and run it through RADISTAT
    SLICE = 66
    img = vol[:, :, SLICE]
    mask = volmask[:, :, SLICE]
    featmap = vol_featmap[:, :, SLICE]

    radistat(img, mask, featmap)

    # TODO visualize results
