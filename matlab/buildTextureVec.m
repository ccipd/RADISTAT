function intensity_vec = buildTextureVec(clusteredImg, binedges, method)
% Function for computing the textural component of RADISTAT

%assumes that clusteredImg ranges from 0 - 1 with NaNs outside ROI

%method = {'prop','wghtprop','propratio'}

fprintf('\tbuilding intensity vec...');
%% TEST 2D IMAGE
% y = [3/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 2/3];
% 
% clusteredImg = y;
% binedges = [0 1/3 2/3 1];

% %% TEST 3D IMAGE
% y(:,:,1) = [1/3 1/3 1/3; 1/3 1/3 1/3; 1/3 1/3 1/3];
% y(:,:,2) = [1/3 1/3 1/3; 1/3 3/3 1/3; 1/3 1/3 1/3];
% y(:,:,3) = [1/3 1/3 1/3; 1/3 1/3 1/3; 2/3 1/3 1/3];
% 
% clusteredImg = y;
% binedges = [0 1/3 2/3 1];

%%
if nargin == 2
    method = 'prop';
end

clustvec = clusteredImg(clusteredImg>0);
e = binedges(2:end);

proportion_vec = zeros(1,3);
intensity_vec = zeros(1,3);

%% prop
if strcmp(method,'prop')
    
    %proportion_vec: proportion of each expression level
    for i = 1:length(e)
        idxs = find(abs(clusteredImg-e(i))<1e-2); %doesn't always work comparing double to double in MATLAB
        if isempty(idxs)
            continue;
        end
        proportion_vec(i) = length(idxs)/length(clustvec);  %vector of histogram bin counts based on distrubtion of clust expression levels
    end   

    intensity_vec = proportion_vec;

%% wghtprop

elseif strcmp(method,'wghtprop')
    
    
    %proportion_vec: proportion of each expression level
    for i = 1:length(e)
        proportion_vec(i) = sum(clustvec==e(i))/length(clustvec);  %vector of histogram bin counts based on distrubtion of clust expression levels
    end   

    %num_vec: proportion of clusters corresponding to each expression level
    num_vec = zeros(1,length(e));
    for i = 1:length(e)
        idxs = find(abs(clusteredImg-e(i))<1e-2); %doesn't always work comparing double to double in MATLAB
        if isempty(idxs)
            continue;
        end
        BW = zeros(size(clusteredImg));
        BW(idxs) = 1;
        BW = logical(BW);
        CC = bwconncomp(BW);
        num_vec(i) = CC.NumObjects;
    end
    num_vec = num_vec ./ sum(num_vec);

    intensity_vec = proportion_vec .* num_vec;

%% propratio
elseif strcmp(method,'propratio')
    
    %proportion_vec: proportion of each expression level
    for i = 1:length(e)
        idxs = find(abs(clusteredImg-e(i))<1e-2); %doesn't always work comparing double to double in MATLAB
        if isempty(idxs)
            continue;
        end
        proportion_vec(i) = length(idxs)/length(clustvec);  %vector of histogram bin counts based on distrubtion of clust expression levels
    end   
    %intensity_vec: ratio of expression levels to one another
    count = 1;
    for i = 1:length(proportion_vec)-1
        for k = i+1:length(proportion_vec)
            intensity_vec(count) = proportion_vec(i)/proportion_vec(k);
            count = count + 1;
        end
    end
    intensity_vec(find(intensity_vec>100)) = 100;
    intensity_vec(find(isnan(intensity_vec)))= 0;

%%
else
    error('Invalid method parameter.')
end

fprintf('done.\n');


% %%  optional: plot feature expression histogram -- not normalized
% figure('Color','white');
% subplot(1,2,2);
% h = bar(intensity_vec,'facecolor','g','edgecolor','k');set(h,'BarWidth',1);
% set(gca,'Ylim',[0 1],'box','off','Fontname','Timesnewroman','Fontsize',12,'XtickLabel',[],'XTickLabel',{'L','M','H'},'Fontsize',12);
% title(['nbins = ' int2str(length(binedges(2:end)))]);xlabel('Feature Expression Levels');
% ylabel('Texture Score');  
% subplot(1,2,1);
% cmap = jet(4);cmap(1,:) = [0 0 0];
% colormap(cmap);caxis([0 1]);
% vv(clusteredImg,[],cmap,[0 1]);
