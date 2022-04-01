%% Example script and files for demonstrating how to compute RADISTAT

% Required things:
% 1- a 2D image or 3D volume
% 2 - a mask / contour corresponding to a region of interest (ROI) on that image
% 3 - a feature map with feature values at pixels inside the mask ROI

% add for necessary paths for subfunctions and files
funcname = 'demo_RADISTAT.m';
funcpath = which(funcname);
funcdir = funcpath(1:end-length(funcname));
addpath(fullfile(funcdir,'demo_files'));
addpath(fullfile(funcdir,'demo_subfunctions'));

%% Patient 1 - let's compute it in 2D

%load in the 3D matrices
vol = mha_read_volume(mha_read_header('ex1_vol.mha'));
volmask = mha_read_volume(mha_read_header('ex1_mask.mha'));
vol_feature_map = mha_read_volume(mha_read_header('ex1_feature_map.mha'));

%unfortuately this particular data I have needs to be rotated
vol = permute(vol,[2 1 3]);
volmask = permute(volmask,[2 1 3]);
vol_feature_map = permute(vol_feature_map,[2 1 3]);

%pick one 2D slice from the matrices
sl = 66;
img = vol(:,:,sl);
mask = volmask(:,:,sl);
feature_map = vol_feature_map(:,:,sl);

options = [];

%pass it into RADISTAT!
[radistat_struct, errormsg] = RADISTAT(img,mask,feature_map,options) ;

%visualize the results
superpixel_labels = radistat_struct.supervoxel_labels;
cluster_values = radistat_struct.cluster_values;
expression_values = radistat_struct.expression_values;
texture_vec = radistat_struct.texture_vec;
spatial_vec = radistat_struct.spatial_vec;

figure;
view_radistat(img,mask,feature_map,superpixel_labels,cluster_values,expression_values,texture_vec,spatial_vec,[],[])

%% Patient 2 - let's compute it in 3D

%load in the 3D matrices
vol = mha_read_volume(mha_read_header('ex2_vol.mha'));
volmask = mha_read_volume(mha_read_header('ex2_mask.mha'));
vol_feature_map = mha_read_volume(mha_read_header('ex2_feature_map.mha'));

%unfortuately this particular data I have ALSO needs to be rotated
vol = permute(vol,[2 1 3]);
volmask = permute(volmask,[2 1 3]);
vol_feature_map = permute(vol_feature_map,[2 1 3]);

options = [];

%pass it into RADISTAT!
[radistat_struct, errormsg] = RADISTAT(vol,volmask,vol_feature_map,options) ;

%visualize the results - note that in matlab the only easy way to visualize
%this is in 2D. I use 3D slicer to visualize it in 3D (steps for that not here).
supervoxel_labels = radistat_struct.supervoxel_labels;
cluster_values = radistat_struct.cluster_values;
expression_values = radistat_struct.expression_values;
texture_vec = radistat_struct.texture_vec;
spatial_vec = radistat_struct.spatial_vec;

figure;
view_radistat(vol,volmask,vol_feature_map,supervoxel_labels,cluster_values,expression_values,texture_vec,spatial_vec,[],[])