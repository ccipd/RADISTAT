function [supervoxel_id, vk, vol_supervoxel] =...
    slic_supervoxels(x, vol_mask, step, num_min_voxel)
% Generates superpixel clusters based on inputted feature vectors

% ==== INPUT ====
% x: the MxP data matrix where M is the # of voxels and P the number of
%       features/variables, in this case the observations/trials.
%
% vol_mask: RxCxD volume matrix filled with zero elements except the roi
%       indicated with positive number. Here we use 1.
%       It is important that # positive elements in vol_mask = M. Indeed, we can
%       show the feature p in x_hat in volume format by
%       vol_mask(vol_mask(:)>0)=x_hat(:,p);
%
% step: the real positive step size when initial the seed. For instance,
%       step = [3 2 5] means that the initial seeds are 3-, 2- and 5- unit away
%       in row, column and depth direction.
%
% num_min_voxel: the minimum number of voxels allowed in a supervoxel. That
%       is, it is not possible to have less than num_min_voxel voxels in each
%       supervoxel.
%
% RECOMMENDED: Input largest connected component in mask (good when
%       there are artifacts in mask).

% ==== OUTPUT ====
% supervoxel_id: The column vector of the size Mx1. The mth entry
%       represents the membership of the voxel m. That is, supervoxel_id(m) =
%    label of m taking a value in 1 to K.
%
% vk: Kx(P+|S|) matrix, the kth row represents the feature vector of kth
%       center. [colormap X Y Z] of kth center, in feature space.
%
% vol_supervoxel: RxCxD volume matrix; the supervoxel_id in original volume format
%

%% -------- Start here -----------

%Previous parameters now with fixed values:

epsilon = 1e-3; %the k-mean will stop when the mean displacement of centers is less than epsilon; the smaller epsilon it is, the longer the k-mean runs.
lambda = [1 1 1 1];
num_max_ite = 100; % the number of maximum iteration

[R, C, D] = size(vol_mask);

% extract row, col, depth from the mask
% % meshgrid(x,y,z) --> meshgrid(c,r,d)
[c,r,d] = meshgrid(1:C,1:R,1:D);
r = r(vol_mask(:)>0);
c = c(vol_mask(:)>0);
d = d(vol_mask(:)>0);

% the spatial features don't need to be z-scored
s = [r, c, d]; % row, column, depth, NOT x,y,z
smax = max(s,[],1);
smin = min(s,[],1);

% the beta features, z-scored
[M,P] = size(x);
x = zscore(x,[],1);

% make a total feature and assign the weight
x_hat = [x s];
% combine the weight from feature, location and user-defined (lambda)
w = [ones(1,size(x,2)),1/R,1/C,1/D].*lambda;

%% ====== start doing k-mean =====
fprintf('\tComputing superpixel clusters...');
% initialize x, y, z centers by slicing the space with the user-defined stepsize
% meshgrid(x,y,z) --> meshgrid(c,r,d)
[c0,r0,d0] = meshgrid(1:step(2):C,1:step(1):R,1:step(3):D);
s0 = [r0(:),c0(:),d0(:)]; % row, column, depth
% trim the initial centers to save computational resources
s0 = s0( s0(:,1)>=smin(1) & s0(:,1)<=smax(1)...
    & s0(:,2)>=smin(2) & s0(:,2)<=smax(2)...
    & s0(:,3)>=smin(3) & s0(:,3)<=smax(3),:);
 if isempty(s0)
     fprintf('Bad seed step size. Unable to initialize centers\n');
     supervoxel_id = [];vk=[];vol_supervoxel=[]; num_voxels_in_supervoxels=[];
     delta_s_final=[]; t_final=[];
     return;
 end

%% Initialize the clusters ck
% 1) get the neighborhoods in voxel space for each center
% 2) compute the feature vector for each ck in feature space
% 3) eliminate isolated centers (if there is any), that is the centers with
% no neighbors
ck = zeros(size(s0,1),size(x_hat,2));
nCk = size(s0,1); % the number of centers
ck_list = ones(size(ck,1),1);
for k = 1:nCk
    %% find neighborhood within 2delta from each center in s0
    nh_idx = getneighbors(s,s0,[step(1) step(2) step(3)],k);
    if isempty(nh_idx)
        ck_list(k) = 0; % masked as isolated
    else
        nhk = x_hat(nh_idx,:);
        ck(k,:) = mean(nhk,1);
    end
end
% eliminate the isolated centers
ck = ck(ck_list==1,:);
nCk = size(ck,1);

%% iterative k-mean
delta_s = inf;
t = 1;
while (delta_s > epsilon) && (t<num_max_ite)
    %% recalculate the seed ck
    
        % calculate the distance from nhk to ck
    dist_ik = inf+zeros(M,nCk); % distance matrix
    % check list to see which center is isolated
    ck_list = ones(size(ck,1),1);
    for k = 1:nCk
        % calculate the nhk
        nh_idx = getneighbors(s,ck(:,[end-2:end]),[step(1) step(2) step(3)],k);
        if isempty(nh_idx) % in fact, I don't need to check...
            ck_list(k) = 0;
        else
            nhk = x_hat(nh_idx,:);
            delta = nhk-repmat(ck(k,:),size(nhk,1),1);
            % the feature needs to reweight because the spatial feature s
            % is *not* standardized!!! So, we will need to scale down when
            % calculate the distance in feature space.
            delta = delta.*repmat(w,size(delta,1),1); % reweight the features
            dist_ik(nh_idx,k) = sum(delta.^2,2); % fill the distance-squared matrix
        end
    end
    %% assign parent nodes to the voxels
    [dist_min,k_star] = min(dist_ik,[],2);
    % k_star: the column vector whose entry holds the center # associated
    % with the voxel
    
    %% assign the spatially-nearest parent to the isolated voxel x
    % isolated voxels are voxels that do not belong to any center. Such a
    % situation can happen when the centers move away from some voxels
    isolated_x_idx = find(dist_min == inf);
    if any(isolated_x_idx)
        for i = 1:length(isolated_x_idx)
            delta = repmat(x_hat(isolated_x_idx(i),end-2:end),nCk,1)-ck(:,end-2:end);
            [~,k_star2] = min(sum(delta.^2,2),[],1);
            k_star(isolated_x_idx(i)) = k_star2;
        end
    end
    % At this point in the code, every voxel has its center# assigned. 
    % HOwever, not all the centers are occupied by a voxel.
    %% check for isolated and unused centers and remove them from the profile
    k_star_alive = unique(k_star);
    
    % keep only the profile of alive centers
    ck = ck(k_star_alive,:);
    dist_ik = dist_ik(:,k_star_alive);
    nCk = size(ck,1);

    %% update the alive center 
    
    % At this point we will need to re-label the center# in k* because some
    % centers are eliminated, so the orignal order of centers is destroyed.
    test_flag1 = 0;
    if test_flag1==1
        % This is an inefficient way to update ck!!!
        % A more efficient way can be done by remove the centers and relabel
        % them without recomputing the centers
        [~,k_star] = min(dist_ik,[],2);
    else
        % This is a more efficient way
        k_star = relabel(k_star);
    end

    
    % at this point it should be clear that all the centers are involved,
    % and there are no isolated centers left
    k_star_alive = unique(k_star);
    if ~isempty(setdiff(k_star_alive',[1:nCk])) % check point
        disp('something''s wrong#1');
    end
    
    % backup the current cluster
    ck_old = ck; 
    
    % update the alive centers
    for k = 1:nCk
        ck(k,:) = mean(x_hat(k_star==k,:),1);
    end

    %% check condition for stop criteria

    % calculate the perturbation distance
    delta = ck(:,end-2:end)-ck_old(:,end-2:end);
    delta_s = mean(sqrt(sum(delta.^2,2)),1);
    %% increment the counter
    t = t+1;
end
%% return delta_s and t
delta_s_final = delta_s;
t_final = t-1;

%% post-process to merge the isolated voxels into a supervoxel
% find disconnected regions in the brains
nCk = size(ck,1);
vol_supx = vol_mask*0;
vol_supx(vol_mask>0) = k_star; % put the k* back into the original volume format

% prepare the volume for non-contiguous regions
vol_regions = vol_supx*0;
region_id = 1; % the id of non-contiguous regions
for i = 1:nCk
    vol_bin = vol_supx*0;
    vol_bin(vol_supx==i)=1;
    % figure(111); clf('reset'); hnd = vol3d('cdata',vol_bin,'texture','3D');
    conn_comp = bwconncomp(vol_bin);
    for r = 1:conn_comp.NumObjects
        vol_regions(conn_comp.PixelIdxList{r}) = region_id;
        region_id = region_id + 1;
    end
    % figure(112); clf('reset'); hnd = vol3d('cdata',vol_regions,'texture','3D');
    
end

% prepare the kr*, the associated discontiguous region label for each voxel
kr_star = vol_regions(vol_mask(:)>0);
% % display the discontiguous regions regions
% slicedisplay(vol_regions);
%% plot the histogram of # voxels per cluster
region_id_list = unique(kr_star(:)');
nRk = length(region_id_list);
num_voxels_in_regions = zeros(nRk,1); % the number of voxels contained in each non-contiguous region 
rk = zeros(nRk,size(x_hat,2)); % the center (in feature space) of each region
% count the number of voxels in each region and calculate the center for
% each region
for i = 1:nRk
    num_voxels_in_regions(i) = sum(kr_star==i);
    rk(i,:) = mean(x_hat(kr_star==i,:),1);
end

%% Assign "insufficient region", a region whose number of voxels < num_min_voxel, to some other regions
% num_min_voxel = 5;
rk_insuff = find(num_voxels_in_regions < num_min_voxel);
% kr_star_trim = kr_star;
for i = 1:length(rk_insuff)
    % populate all the voxels in each insufficient region
    s0_rk_isol = s(kr_star==rk_insuff(i),:);
    
    % find all the neighborhoods
    % union all the neighbors together
    nhkr_insuff = [];
    for j = 1:size(s0_rk_isol,1)
        nhkr = getneighbors(s,s0_rk_isol,[1 1 1],j);
        nhkr_insuff = union(nhkr_insuff,nhkr);
    end
    % exclude the voxels originally in the insufficient region out of the
    % neighborhood
    nhkr_insuff = setdiff(nhkr_insuff,find(kr_star==rk_insuff(i)));
    kr_candidates = kr_star(nhkr_insuff(:));
    try
        % check the size of neighborhood supervoxels
        idx_valid_nh = num_voxels_in_regions(kr_candidates)>=(num_min_voxel-size(s0_rk_isol,1));  
        if any(idx_valid_nh)
           [kr_star_mode,kr_star_freq] = mode(kr_candidates(idx_valid_nh),1);
        else
            [kr_star_mode,kr_star_freq] = mode(kr_candidates,1);
        end
        %kr_star_trim(kr_star==rk_insuff(i)) = kr_star_mode;
        kr_star(kr_star==rk_insuff(i)) = kr_star_mode;
    catch %prevent "Index exceeds matrix dimensions" error
        %??
        %just keep default kr_star for that voxel
    end
    
end

%% Relabel the voxel so that the label starts from 1
% supervoxel_id = relabel(kr_star_trim);
supervoxel_id = relabel(kr_star);
vol_supervoxel = vol_mask*0; 
vol_supervoxel(vol_mask>0) = supervoxel_id;
supervoxel_id_list = unique(supervoxel_id(:)');
num_supervoxel = length(supervoxel_id_list);
% recalculate the centers in feature space
vk = zeros(num_supervoxel,size(x_hat,2));
num_voxels_in_supervoxels = zeros(num_supervoxel,1);
for i = 1:num_supervoxel
    vk(i,:) = mean(x_hat(supervoxel_id==supervoxel_id_list(i),:),1);
    num_voxels_in_supervoxels(i) = sum(supervoxel_id==i);
end

fprintf('done.\n');

%*****SUBFUNCTIONS******%
function nh_idx = getneighbors(s,s0,ds,k)
% This function find the neighborhood points from s, for every point in s0.
% However, this version of the code is for 3D only.
%
% INPUT
% s: m x 3 matrix, each row corresponding to a point, the first, second,
% third columns are row, column and depth respectively. s is the set of
% points from which we want to search for the neighbors.
%
% s0: m0 x 3 matrix, each row corresponding to a point we hope to find the
% neighbor for.
%
% ds: The step size in row, column and depth
% 
% k: an iterator needed to prevent use of cell arrays; so only returns
% neighbors for one instance

% OUTPUT
% nh_idx: Contains list of indices of the neighbors in s.

%% Get the stepsize in each dimension
dx = ds(1); dy = ds(2); dz = ds(3);
[m, d] = size(s);
[m0, d0] = size(s0);

% check if the dimensions are equal
if d~=d0
    nh_idx = [];
    return;
end

% if the dimensions are equal

    tmp = s(:,1)>=s0(k,1)-dx & s(:,2)>=s0(k,2)-dy & s(:,3)>=s0(k,3)-dz ...
        & s(:,1)<=s0(k,1)+dx & s(:,2)<=s0(k,2)+dy & s(:,3)<=s0(k,3)+dz;
    nh_idx = find(tmp'==1);

return;

function [B, label_org] = relabel(A)
%relabel rearranges matrix entries according to its rank order, so that the
%smallest number will start with rank 1, and so on
%
% ===== Example ====
% X = [-1.2000   -1.2000   20.3200    5.0000    5.0000
%      20.3200    5.0000    8.0000    5.0000    5.0000
%      20.3200    8.0000   20.3200    5.0000    8.0000
%      20.3200   20.3200   -1.2000    5.0000    8.0000
%       8.0000   20.3200    8.0000    8.0000    8.0000]
% 
% [B,map] = relabel(X);
% 
% B =
% 
%      1     1     4     2     2
%      4     2     3     2     2
%      4     3     4     2     3
%      4     4     1     2     3
%      3     4     3     3     3
% 
% map =
% 
%    -1.2000
%     5.0000
%     8.0000
%    20.3200     
%      
% ======= mapping =====
%    -1.2000 ---> 1
%     5.0000 ---> 2
%     8.0000 ---> 3
%    20.3200 ---> 4 

%% === code content ====

label_org = unique(A(:));
B = A;
for i = 1:length(label_org)
    B(A==label_org(i)) = i;
end


    