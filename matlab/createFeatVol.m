function featvol = createFeatVol(featvec, mask)
% Function for generating 3D volume of feature data within mask ROI

%INPUTS
%   featvec = a Mx1 vector (cell or double) containing the feature intensities
%               for M voxels in the corresponding mask
%      mask = a 3D logical matrix correndponding to an annotated
%             region of interest (***label == 1***).
%   rescale = (optional) user may wish to have the featvec intensities
%                   rescaled. 'rescale' must be provided. Default is no 
%                   rescale. If 'rescale' inputted, a 4th parameter may
%                   provided to specify the range
%      range = (optional) 2D vector of rescale range. [min max]. Default is
%               [0 1].

%OUTPUTS
%   featvol = a 3D matrix the same size as mask. Featvec intensities are
%               inserted into the nonzero mask voxels, and background voxels are
%               assigned "nan" values.

%reformat feature vector if cell input
% if isa(featvec,'cell')
%     featvec = cell2mat(featvec);
% end

% typecast
featvec = double(featvec);
mask = double(mask);

%check input params
if length(featvec) ~= length(find(mask>0))
    error('Number of voxels in featvec must be equal to the number of voxels in the masked region of interest');
end

%rescale feature intensities
% if nargin >= 3
%     if strcmp(rescale,'rescale')
%         if nargin == 3
%             range = [0 1];
%         end
%         featvec = rescale_range(featvec,range(1),range(2));
%     end
% end

%create feature volume
featvol = NaN(size(mask));
featvol(mask==1) = featvec;

end
