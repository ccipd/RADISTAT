function [cluster_values,errormsg] = cluster_checkpoint(mask, feature_values, supervoxel_labels)
% Reassign the value of each cluster to the mean value of all pixels in the cluster.

%% CHECK INPUT PARAMETERS
if (length(find(mask>0))~=length(feature_values))
    error('check MASK or FEATURE_VALUES -- not the same number of pixels');
end

cluster_values = [];
errormsg = [];

%% CHECKPOINT: SUPERVOXELS COMPILED
    
% Bad supervoxels?
if isempty(supervoxel_labels)
    errormsg = 'Unable to generate supervoxels.';
    return;
else
    if ~isempty(find(isnan(supervoxel_labels),1))
        errormsg = 'Nan assigned to supervoxel cluster.';
        return;
    elseif (length(feature_values)~=length(supervoxel_labels))
        error('check FEATURE_VALUES or SUPERVOXEL_LABELS -- not the same number of pixels');
        return;
    end
end
                
% Otherwise, let's fill clusters with mean cluster value
cluster_values = zeros(size(feature_values));
u = unique(supervoxel_labels);
for i = 1:length(u)
    cluster_values(supervoxel_labels==u(i)) = mean(feature_values(supervoxel_labels==u(i)));
end