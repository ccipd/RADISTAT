function [expression_values,errormsg] = expression_checkpoint(cluster_values, options)
% Assigns RADISTAT expression values to each superpixel cluster

%% CHECK INPUT PARAMETERS


if exist('options','var')
    if ~isfield(options,'percvalues')
        error('OPTIONS struct missing field PERCVALUES.');
    elseif ~isfield(options,'nbins')
        error('OPTIONS struct missing field NBINS.');
    elseif (options.nbins - length(options.percvalues)) ~= 1
        error('OPTIONS.NBINS must be 1 more than length of OPTIONS.PERCVALUES.')
    end
else %defaults
    options.percvalues(1) = prctile(cluster_values,33); %threshold 1: 33rd prctile
    options.percvalues(2) = prctile(cluster_values,67); %threshold 2: 67rd prctile
    options.nbins = 3; %Low, Medium, High expression levels
end

expression_values = [];
errormsg = [];

%% CHECKPOINT: ALL EXPRESSION LEVELS PRESENT
fprintf('\tRe-partitioning clusters into RADISTAT expression levels...');

% rebin
edges = 0:1/options.nbins:1; 

expression_values = cluster_values; %initialize
x = -inf; %initialize

% reassign cluster values based on user-defined thresholds
for j = 1:length(options.percvalues) 
    x = [x options.percvalues(j)];
    checknum(j) = length(find(cluster_values>x(j) & cluster_values<=x(j+1)));
    expression_values(cluster_values>x(j) & cluster_values<=x(j+1)) = edges(j+1);
end
checknum(j+1) = length(find(cluster_values>x(j+1)));
expression_values(cluster_values>x(j+1)) = edges(end);
if ~all(checknum)
    errormsg = 'At least one expression level is missing.';
end

fprintf('done.\n');
