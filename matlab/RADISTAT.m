function [radistat_struct, errormsg] = RADISTAT(img,mask,feature_map,options) 
% MAIN function for RADISTAT implementation

% INPUTS
% img: double M x N x S matrix containing image from which 
%      feature values were extracted
% mask: double M x N x S matrix corresponding to label on img
%       within which featvec were extracted
% feature_map: double M x N x S matrix containing the representative 
%                 feature values for each nonzero pixel in mask ROI
% options: (optional) struct with fields specifying parameters for RADISTAT
%      .percvalues: double 1 x R array of feature values at user-desired thresholds
%                 (i.e. the feature values at the 33rd and 67th percentile)
%      .ws: double 1 x 1 specifying the window size of supervoxels to be
%           used (i.e. ws=3, 5, 7, 9, etc.)
%      .num_min_voxel: double 1 x 1 specifying minimum number of voxels in
%                      a supervoxel cluster (i.e. num_min_voxel = 10, 25, 50, etc.)
%      .texture_metric: string specifying which texture metric should be
%                      computed ('prop','propratio','wghtprop')
%      .spatial_metric: string specifying which spatial metric should be
%                      computed ('adjacency')
%      .nbins: double 1 x 1 indicating number of expression levels to be used
%      .view: string specifying whether to display results ('on','off')

% OUTPUTS
% radistat_struct: struct with fields of various metrics computed in RADISTAT
%          .supervoxel_labels = double Q x 1 vector of labels specifying
%                               which supervoxel cluster each element in 
%                               featvec was assigned to
%          .cluster_values = double Q x 1 vector of new feature values of
%                            each element in supervoxel clusters
%          .expression_values = double Q x 1 vector or assigned expression
%                               levels of each element in supervoxel clusters
%          .texture_vec = double 1 x T array of texture metric values for each bin
%          .spatial_vec = double 1 x S array of spatial metric values for each bin
% errormsg = string which will indicate if any issues occured which may
%            negatively impact or inhibit compiliation of RADISTAT

%% 0. Check necessary paths/files exist
if exist('slic_supervoxels.m','file')~=2
    error('Get slice_supervoxels.m into your path.');
end

if exist('createFeatVol.m','file')~=2
    error('Get createFeatVol.m into your path.');
end

if exist('cluster_checkpoint.m','file')~=2
    error('Get cluster_checkpoint.m into your path.');
end

if exist('expression_checkpoint.m','file')~=2
    error('Get expression_checkpoint.m into your path.');
end

if exist('buildSpatialVec.m','file')~=2
    error('Get buildSpatialVec.m into your path.');
end

if exist('buildTextureVec.m','file')~=2
    error('Get buildTextureVec.m into your path.');
end


%% 1. Check Inputs

radistat_struct = [];
errormsg = [];

if ~all(size(img)==size(mask))
    error('IMG and MASK must be same size.');
end

if ~all(size(mask)==size(feature_map))
    error('MASK and FEATURE_MAP must be same size.');
end
    
if isempty(find(mask>0))
   errormsg = 'No nonzero MASK values. Cannot compute RADISTAT.';
   return;
end

feature_values = feature_map(mask>0);
if (length(find(mask>0))~=length(feature_values))
    error('check MASK or FEATURE_VALUES -- not the same number of pixels');
end

if exist('options','var') || isempty(options) %check for any missing fields. fill them in
    if ~isfield(options,'percvalues')
        options.percvalues(1) = prctile(feature_values,33); %threshold 1: 33rd prctile
        options.percvalues(2) = prctile(feature_values,67); %threshold 2: 67rd prctile
    end
    if ~isfield(options,'texture_metric'), options.texture_metric = 'prop'; end
    if ~isfield(options,'spatial_metric'), options.spatial_metric = 'adjacency'; end
    if ~isfield(options,'nbins'), options.nbins = 3; end
    if ~isfield(options,'ws'), options.ws = 5; end
    if ~isfield(options,'num_min_voxel'), options.num_min_voxel = 5; end
    if ~isfield(options,'view'), options.view = 'off'; end
else %defaults
    options.percvalues(1) = prctile(feature_values,33); %threshold 1: 33rd prctile
    options.percvalues(2) = prctile(feature_values,67); %threshold 2: 67rd prctile
    options.ws = 5;
    options.num_min_voxel = 5; 
    options.texture_metric = 'prop'; %prop, propratio, or wghtprop
    options.spatial_metric = 'adjacency'; %adacency
    options.nbins = 3; %Low, Medium, High expression levels
    options.view = 'off';
end

if (options.nbins - length(options.percvalues)) ~= 1
        error('OPTIONS.NBINS must be 1 more than length of OPTIONS.PERCVALUES.')
elseif options.nbins > 3
        error('Current implemenation of RADISTAT only allows for OPTIONS.NBINS = 3');
end

%% 2a. Extract Superpixels

if numel(size(img))==2 %2D
    step = [options.ws options.ws 1];
    %** supervoxel code only works for 3D inputs, so lets just replicate the image and mask**
    [~, ~, vol_supervoxel] = slic_supervoxels(repmat(feature_values,[2 1]), repmat(mask,[1 1 2]), step, options.num_min_voxel);
    vol_supervoxel = vol_supervoxel(:,:,1);
    
elseif numel(size(img))==3 %3D
    step = [options.ws options.ws options.ws];
    [~,~,vol_supervoxel] = slic_supervoxels(feature_values,mask,step,options.num_min_voxel);
end

supervoxel_labels = vol_supervoxel(mask>0);

    
%% 2b. Reassign Cluster Values
[cluster_values,errormsg] = cluster_checkpoint(mask, feature_values, supervoxel_labels);
if ~isempty(errormsg)
    return;
end
%% 3. Assign Expression Levels
[expression_values,errormsg] = expression_checkpoint(cluster_values, options);

%we will still let it extract RADISTAT in case we need it

%% 4. Extract RADISTAT vectors

f = createFeatVol(expression_values,mask);
% [f,~] = boundingbox2(f,mask,5,'allz','off');

edges = 0:1/options.nbins:1; 

% Calculate Intensity Metrics
texture_vec = buildTextureVec(f, edges,options.texture_metric);       

% Calculate Spatial Metrics
spatial_vec = buildSpatialVec(f,edges,options.spatial_metric);
          
%%
radistat_struct.supervoxel_labels = supervoxel_labels;
radistat_struct.cluster_values = cluster_values;    
radistat_struct.expression_values = expression_values;    
radistat_struct.texture_vec = texture_vec;
radistat_struct.spatial_vec = spatial_vec;

fprintf('done.\n');