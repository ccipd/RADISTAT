function ploth = display_feature_map(img,mask,featvec)
% Function for visualizing feature heatmaps overlaid onto imaging data

% INPUTS
%       1- img: 2D slice or 3D imgume of interest
%       2- mask: corresponding label mask for img
%       3- featvec: 1D vector of feature values extracted within mask
% OUTPUTS
%       1- ploth: handle to current plot

% example call:
%   ploth = feature_map(img,mask,featints)


%% check inputs
if nargin ~= 3
    error('Incorrect number of arguments');
end

if ~all(ismember(size(img),size(mask)))
    error('Size of MASK must equal size of IMG');
end
if length(featvec)~=length(find(mask>0))
    error('Length of FEATINTS must equal nonzero pixels in MASK');
end
if numel(size(img))~=2 
    error('FEATURE_MAP_SLICE.M handles 2D images only. See FEATURE_MAP.M for 3D visualization.');
end

%% Preapre Data

featslice = createFeatVol(featvec,mask);

%comment/uncomment as necessary
img = double((img));
mask = double((mask));
featslice = double((featslice));

%% ----overlay of feature map onto img------%
cla;colorbar off;

imgslice = img/max(img(:));
rgbslice = imgslice(:,:,[1 1 1]);

bwROIlocations = ~isnan(featslice);
g = imagesc(featslice);
colormap(gca,'jet'); c = colorbar('east');
c.Color = 'w'; c.FontSize = 12;

alpha(g,1); 
hold on; h = imagesc(rgbslice);
set(h,'AlphaData',~bwROIlocations);
axis image
axis off

if nargout == 1
    ploth = h;
end

end



